# scoring_cluster_parallel.py
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from seq_features import all_params, net_charge, mean_hydrophobicity, pro_gly_fraction
from motif_agg_features import (
    find_gxsxg,
    window_charge,
    max_hydrophobic_window,
    max_hydrophobic_run,
)
from alignment import main_process as blossum_align
from ph_optimum import main_process as phOpt
from km_kcat_pred import main_process as kinetic_pred

# ForgeESM embedder and wt_mapping utilities (from repo)
from forge_esm_scoring import ForgeESMEmbedder, pool_mean
from wt_mapping import build_cluster_to_wt_map


# ==============================================================================
# HELPERS
# ==============================================================================

def zscore(x: np.ndarray) -> np.ndarray:
    """Compute z-score normalization."""
    mu = x.mean()
    sd = x.std() if x.std() > 0 else 1.0
    return (x - mu) / sd


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (safe, L2-normalized)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ==============================================================================
# STEP 0 — WT MAPPING
# Builds cluster_to_wt.csv and sequence_to_wt.csv from test + WT CSVs.
# This MUST run before ESM embedding so every sequence has a WT reference.
# ==============================================================================

def build_wt_mapping(
    test_csv: str,
    wt_csv: str,
    clusters_csv: str,
    out_cluster_map: str = "cluster_to_wt.csv",
    out_seq_map: str = "sequence_to_wt.csv",
    seq_col: str = "sequence",
    k: int = 3,
    dim: int = 4096,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper around wt_mapping.build_cluster_to_wt_map.

    Produces:
      cluster_to_wt.csv  → cluster_id | wt_id | wt_sequence | wt_centrality | wt_cds
      sequence_to_wt.csv → sequence   | cluster_id | wt_id | wt_sequence

    The WT for each cluster is chosen by:
      1) Finding which known WT sequences appear in the test dataset
      2) Assigning each cluster its most "central" WT via k-mer cosine similarity
         (see wt_mapping.choose_cluster_wt)

    Args:
        test_csv:         Path to input sequences CSV
        wt_csv:           Path to wildtype reference CSV (must have 'Wt AA Sequence' column)
        clusters_csv:     Path to backbone_clusters.csv (sequence, cluster_id)
        out_cluster_map:  Output path for cluster → WT mapping
        out_seq_map:      Output path for sequence → WT mapping
        seq_col:          Column name in test_csv that holds sequences
        k:                k-mer size for centrality scoring
        dim:              Hash dimension for k-mer vectors

    Returns:
        (cluster_map_df, sequence_map_df)
    """
    print("Building WT mapping per cluster...")
    cluster_map, seq_map = build_cluster_to_wt_map(
        test_csv=test_csv,
        wt_csv=wt_csv,
        backbone_clusters_csv=clusters_csv,
        out_cluster_map_csv=out_cluster_map,
        out_sequence_map_csv=out_seq_map,
        seq_col=seq_col,
        k=k,
        dim=dim,
    )
    n_clusters = len(cluster_map)
    n_unmapped = (cluster_map["wt_id"] < 0).sum()
    print(f"  → {n_clusters} clusters | {n_unmapped} clusters without WT mapped")
    print(f"  → Saved: {out_cluster_map}, {out_seq_map}")
    return cluster_map, seq_map


# ==============================================================================
# STEP 1a — CLUSTER + WT MERGE
# Replaces old add_clusters() (which used cluster_representatives.csv).
# Now uses sequence_to_wt.csv to get the per-cluster WT reference sequence.
# ==============================================================================

def add_clusters_with_wt(
    df: pd.DataFrame,
    clusters_csv: str,
    seq_to_wt_csv: str,
    col_seq: str = "sequence",
    wt_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge sequences with cluster_id and their matched wt_sequence.

    Follows compute_delta_forge_vs_wt.py logic exactly:
      1) test sequences → cluster_id         (via backbone_clusters.csv)
      2) cluster_id    → wt_sequence         (via sequence_to_wt.csv, deduped per cluster)
      3) Fallback for unmapped sequences:
           a) Length-matched WT from wt_csv  (matches compute_delta_forge_vs_wt.py fallback)
           b) Self as WT                     (last resort; delta_emb will be ~0)

    Args:
        df:             DataFrame with sequence column
        clusters_csv:   backbone_clusters.csv (columns: sequence, cluster_id)
        seq_to_wt_csv:  sequence_to_wt.csv   (columns: sequence, cluster_id, wt_id, wt_sequence)
        col_seq:        Name of sequence column in df
        wt_csv:         Optional path to WT CSV for length-based fallback
                        (must have 'Wt AA Sequence' column)

    Returns:
        df with added columns: cluster_id, wt_sequence
    """
    cl_df  = pd.read_csv(clusters_csv)
    map_df = pd.read_csv(seq_to_wt_csv)

    # 1) sequence → cluster_id
    df = df.merge(
        cl_df[["sequence", "cluster_id"]].rename(columns={"sequence": col_seq}),
        on=col_seq,
        how="left",
    )

    missing_cluster = df["cluster_id"].isna()
    if missing_cluster.any():
        n = int(missing_cluster.sum())
        print(f"  WARNING: {n} sequences have no cluster_id — assigning -1")
        df.loc[missing_cluster, "cluster_id"] = -1

    df["cluster_id"] = df["cluster_id"].astype(int)

    # 2) cluster_id → wt_sequence
    # Dedupe: keep one wt_sequence per cluster_id to avoid cartesian explosion on merge
    map_unique = (
        map_df[["cluster_id", "wt_sequence"]]
        .dropna(subset=["wt_sequence"])
        .drop_duplicates(subset=["cluster_id"])
        .copy()
    )
    df = df.merge(map_unique, on="cluster_id", how="left")

    # 3) Fallback for sequences still missing a WT
    missing_wt = df["wt_sequence"].isna() | (df["wt_sequence"].astype(str).str.strip() == "")
    if missing_wt.any():
        n = int(missing_wt.sum())
        print(f"  WARNING: {n} sequences have no WT mapping.")

        if wt_csv is not None and os.path.exists(wt_csv):
            # Length-based fallback (mirrors compute_delta_forge_vs_wt.py)
            wt_df = pd.read_csv(wt_csv)
            wt_df["wt_len"] = wt_df["Wt AA Sequence"].astype(str).str.len()
            wt_by_len = (
                wt_df.sort_values("wt_len")
                .groupby("wt_len")["Wt AA Sequence"]
                .first()
                .to_dict()
            )
            wt_global = str(wt_df["Wt AA Sequence"].iloc[0])

            if "seq_len" not in df.columns:
                df["seq_len"] = df[col_seq].astype(str).str.len()

            def fill_wt_by_len(row):
                L = int(row["seq_len"])
                if L in wt_by_len:
                    return wt_by_len[L]
                closest = min(wt_by_len.keys(), key=lambda x: abs(x - L))
                return wt_by_len.get(closest, wt_global)

            df.loc[missing_wt, "wt_sequence"] = df.loc[missing_wt].apply(fill_wt_by_len, axis=1)
            print(f"  → Filled {n} sequences with length-matched WT fallback.")
        else:
            # Absolute last resort: use self (delta_emb ≈ 0, loses signal)
            print("  → No wt_csv provided. Using self as WT (delta_emb will be ~0).")
            df.loc[missing_wt, "wt_sequence"] = df.loc[missing_wt, col_seq]

    return df


# ==============================================================================
# STEP 1b — ESM EMBEDDING + COSINE DELTA
#
# WHAT CHANGED vs original code:
#   Old: esm_ll_for_sequences(seqs) → scalar log-likelihood per sequence
#        delta_esm = esm_ll_seq - esm_ll_rep  (rep = cluster centroid seq)
#
#   New: ForgeESMEmbedder.embed_one(seq) → mean-pooled embedding vector (D,)
#        cosine_sim = dot(pool_mean(embed(seq)), pool_mean(embed(wt_sequence)))
#        delta_emb  = 1 - cosine_sim
#             → 0.0 means sequence is identical to WT in embedding space
#             → 1.0 means maximally diverged from WT
#
# WHY: compute_delta_forge_vs_wt.py uses ESM3 Forge API (esmc-6b-2024-12),
# not local ESM log-likelihood. The correct WT reference is the per-cluster WT
# (from wt_mapping), NOT the cluster representative sequence.
# ==============================================================================

def compute_esm_delta_embeddings(
    df: pd.DataFrame,
    col_seq: str = "sequence",
    col_wt: str = "wt_sequence",
    esm_model: str = "esmc-6b-2024-12",
    esm_cache_dir: str = "cache_forge_esmc",
) -> pd.DataFrame:
    """
    Compute ESM3 Forge mean-pooled embeddings and cosine similarity
    between each variant and its cluster's WT.

    Mirrors compute_delta_forge_vs_wt.py exactly:
      - Embeds unique WTs first (avoids redundant API calls)
      - Embeds each variant sequence (disk-cached via ForgeESMEmbedder)
      - wt_cosine_sim = cosine(pool_mean(embed(seq)), pool_mean(embed(wt)))
      - delta_emb     = 1 - wt_cosine_sim

    Requires ESM_API_KEY environment variable.

    Args:
        df:             DataFrame with col_seq and col_wt columns
        col_seq:        Variant sequence column name
        col_wt:         WT sequence column name (from add_clusters_with_wt)
        esm_model:      ESM Forge model string
        esm_cache_dir:  Directory for caching embeddings (avoids re-computing on re-runs)

    Returns:
        df with added columns: wt_cosine_sim, delta_emb
    """
    token = os.environ.get("ESM_API_KEY")
    if not token:
        raise EnvironmentError(
            "ESM_API_KEY environment variable not set.\n"
            "Set it with: export ESM_API_KEY='your_forge_token'\n"
            "Get a token at: https://forge.evolutionaryscale.ai"
        )

    embedder = ForgeESMEmbedder(
        model=esm_model,
        token=token,
        cache_dir=esm_cache_dir,
    )

    # Pre-compute pooled vectors for all unique WTs
    wt_seqs_unique = df[col_wt].astype(str).unique().tolist()
    print(f"  Embedding {len(wt_seqs_unique)} unique WT sequences...")
    wt_vecs: Dict[str, np.ndarray] = {}
    for wt in tqdm(wt_seqs_unique, desc="WT embeddings"):
        emb = embedder.embed_one(wt)
        wt_vecs[wt] = pool_mean(emb)

    # Compute per-sequence cosine similarity vs its WT
    print(f"  Embedding {len(df)} variant sequences...")
    wt_cos_list: List[float] = []
    delta_emb_list: List[float] = []

    for seq, wt in tqdm(
        zip(df[col_seq].astype(str), df[col_wt].astype(str)),
        total=len(df),
        desc="Variant embeddings",
    ):
        emb = embedder.embed_one(seq)
        vec = pool_mean(emb)
        sim = cosine_sim(vec, wt_vecs[wt])
        wt_cos_list.append(sim)
        delta_emb_list.append(1.0 - sim)

    df = df.copy()
    df["wt_cosine_sim"] = wt_cos_list
    df["delta_emb"]     = delta_emb_list

    return df


# ==============================================================================
# MOTIF / AGGREGATION FEATURE BATCHES (unchanged)
# ==============================================================================

def compute_motif_features_batch(
    sequences: List[str], start_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(sequences)
    motif_present = np.zeros(n, dtype=np.int32)
    motif_center  = np.full(n, -1, dtype=np.int32)
    local_c55     = np.zeros(n, dtype=np.float32)
    local_c90     = np.zeros(n, dtype=np.float32)
    LOCAL_WIN = 41

    for i, seq in enumerate(sequences):
        mp_flag, mc = find_gxsxg(seq)
        motif_present[i] = mp_flag
        motif_center[i]  = mc
        local_c55[i] = window_charge(seq, mc, LOCAL_WIN, net_charge, 5.5)
        local_c90[i] = window_charge(seq, mc, LOCAL_WIN, net_charge, 9.0)

    return motif_present, motif_center, local_c55, local_c90


def compute_aggregation_features_batch(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    agg_win = np.array([max_hydrophobic_window(s, win=9) for s in sequences], dtype=np.float32)
    agg_run = np.array([max_hydrophobic_run(s) for s in sequences], dtype=np.float32)
    return agg_win, agg_run


# ==============================================================================
# MAIN PROPERTY COMPUTATION
# ==============================================================================

def compute_properties(
    df: pd.DataFrame,
    clusters_csv: str = "backbone_clusters.csv",
    seq_to_wt_csv: str = "sequence_to_wt.csv",
    wt_csv: Optional[str] = None,
    out_path: str = "sequences_raw_properties.csv",
    n_jobs: int = -1,
    batch_size: int = 1000,
    col_seq: str = "sequence",
    esm_model: str = "esmc-6b-2024-12",
    esm_cache_dir: str = "cache_forge_esmc",
) -> pd.DataFrame:
    """
    Compute all sequence properties with parallel processing.

    Cluster + ESM logic follows compute_delta_forge_vs_wt.py:
      - cluster_id + wt_sequence per variant from sequence_to_wt.csv
      - ESM: mean-pooled Forge ESM3 embeddings (cached to disk)
      - delta_emb = 1 - cosine_sim(embed(seq), embed(wt))

    Args:
        df:             Input DataFrame (must contain col_seq column)
        clusters_csv:   backbone_clusters.csv  (sequence, cluster_id)
        seq_to_wt_csv:  sequence_to_wt.csv    (sequence, cluster_id, wt_id, wt_sequence)
        wt_csv:         WT CSV for fallback    ('Wt AA Sequence' column required)
        out_path:       Output CSV path
        n_jobs:         Parallel workers (-1 = all cores)
        batch_size:     Batch size for motif/aggregation parallel steps
        col_seq:        Sequence column name in df
        esm_model:      ESM Forge model string
        esm_cache_dir:  Cache dir for ESM embeddings (safe to reuse across runs)
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    print(f"Computing properties with {n_jobs} cores...")

    df = df.copy()
    df[col_seq] = df[col_seq].astype(str).str.strip()
    df["seq_len"] = df[col_seq].str.len()

    # ── 1. Cluster merge: sequence → cluster_id + wt_sequence ──────────────
    print("\n[1/6] Merging cluster and WT mapping...")
    df = add_clusters_with_wt(
        df,
        clusters_csv=clusters_csv,
        seq_to_wt_csv=seq_to_wt_csv,
        col_seq=col_seq,
        wt_csv=wt_csv,
    )
    print(f"  Unique clusters : {df['cluster_id'].nunique()}")
    print(f"  Unique WTs used : {df['wt_sequence'].nunique()}")

    # ── 2. ESM embedding + cosine delta ────────────────────────────────────
    print("\n[2/6] Computing ESM embeddings and delta_emb...")
    df = compute_esm_delta_embeddings(
        df,
        col_seq=col_seq,
        col_wt="wt_sequence",
        esm_model=esm_model,
        esm_cache_dir=esm_cache_dir,
    )

    # ── 3. Biophysical features ─────────────────────────────────────────────
    print("\n[3/6] Computing biophysical features...")
    seqs = df[col_seq].tolist()

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        protein_params = list(tqdm(
            executor.map(all_params, seqs),
            total=len(seqs),
            desc="Biophysical",
        ))

    protein_params_df = pd.DataFrame(protein_params)
    df = pd.concat([df.reset_index(drop=True), protein_params_df], axis=1)

    # ── 4. Motif & local charge features ───────────────────────────────────
    print("\n[4/6] Computing motif and local charge features...")
    n_seqs = len(seqs)
    batches = [(seqs[i:i + batch_size], i) for i in range(0, n_seqs, batch_size)]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(compute_motif_features_batch, batch, start): idx
            for idx, (batch, start) in enumerate(batches)
        }
        ordered = [None] * len(batches)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Motif"):
            ordered[futures[future]] = future.result()

    df["motif_present"]   = np.concatenate([r[0] for r in ordered])
    df["motif_center"]    = np.concatenate([r[1] for r in ordered])
    df["local_charge_55"] = np.concatenate([r[2] for r in ordered])
    df["local_charge_90"] = np.concatenate([r[3] for r in ordered])

    # ── 5. Aggregation features ─────────────────────────────────────────────
    print("\n[5/6] Computing aggregation features...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(compute_aggregation_features_batch, seqs[i:i + batch_size]): idx
            for idx, i in enumerate(range(0, n_seqs, batch_size))
        }
        ordered_agg = [None] * len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregation"):
            ordered_agg[futures[future]] = future.result()

    df["agg_win9"] = np.concatenate([r[0] for r in ordered_agg])
    df["agg_run"]  = np.concatenate([r[1] for r in ordered_agg])

    # ── 6. Penalties + normalization ────────────────────────────────────────
    print("\n[6/6] Computing penalties and normalizing...")

    mode_len = int(df["seq_len"].mode().iloc[0])
    df["len_pen"]      = -np.abs(df["seq_len"].values - mode_len) / mode_len
    df["motif_penalty"] = 1.0 - df["motif_present"].values.astype(np.float32)

    col_to_normalize = [
        # ESM embedding-based (replaces old delta_esm / esm_ll)
        "delta_emb",
        "wt_cosine_sim",
        # Sequence features
        "len_pen",
        "local_charge_55",
        "local_charge_90",
        "agg_win9",
        "agg_run",
        # Biophysical
        "IEP",
        "charge7",
        "charge9",
        "charge55",
        "molecular_weight",
        "instability_index",
        "hydrophobicity",
        "pg_fraction",
    ]
    # Guard: only normalize columns that exist
    col_to_normalize = [c for c in col_to_normalize if c in df.columns]
    col_normalized   = ["z_" + c for c in col_to_normalize]
    df[col_normalized] = df[col_to_normalize].apply(zscore)

    print(f"\nSaving to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Properties computation complete!")

    return df


# ==============================================================================
# ALIGNMENT / pH / KINETICS
# ==============================================================================

def alignment_blossum(wildtype: str, mutation: str, output: str, n_jobs: int = -1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    print(f"Running BLOSUM alignment with {n_jobs} cores...")
    blossum_align(wildtype, mutation, n_jobs, "loky", output)
    print("BLOSUM alignment complete!")


def compute_ph_optimum_pred(input_csv: str):
    output = input_csv.replace(".csv", "_pH_opt.csv")
    print(f"Computing pH optimum for {input_csv}...")
    phOpt(input_csv, output, "pHopt")
    print(f"pH optimum saved → {output}")


def compute_kcat_km(df: pd.DataFrame, col_seq: str, out_path: str):
    print(f"Computing kcat/km predictions from column '{col_seq}'...")
    kinetic_pred(df[col_seq].tolist(), out_path)
    print("Kinetic predictions complete!")


# ==============================================================================
# PIPELINE CLASS
# ==============================================================================

class ParallelPipeline:
    """
    Full pipeline orchestrator.

    Step 0  — Build WT mapping  (cluster_to_wt.csv + sequence_to_wt.csv)
               Skipped if files exist; set rebuild_wt_map=True to force.
    Step 1  — compute_properties()
                ├─ add_clusters_with_wt()          sequence → cluster_id + wt_sequence
                ├─ compute_esm_delta_embeddings()   cosine delta vs WT in embedding space
                ├─ biophysical features             (parallel threads)
                ├─ motif + local charge             (parallel processes)
                ├─ aggregation features             (parallel processes)
                └─ z-score normalization
    Step 2  — BLOSUM alignment
    Step 3+4 — pH optimum + kcat/km  (parallel threads)
    """

    def __init__(
        self,
        n_jobs: int = -1,
        esm_model: str = "esmc-6b-2024-12",
        esm_cache_dir: str = "cache_forge_esmc",
        rebuild_wt_map: bool = False,
    ):
        """
        Args:
            n_jobs:          Parallel workers (-1 = all cores)
            esm_model:       ESM Forge model name
            esm_cache_dir:   Embedding cache dir (reused across runs — saves API calls)
            rebuild_wt_map:  Force rebuild of cluster_to_wt / sequence_to_wt even if they exist
        """
        self.n_jobs        = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.esm_model     = esm_model
        self.esm_cache_dir = esm_cache_dir
        self.rebuild_wt_map = rebuild_wt_map
        print(f"Initialized pipeline | cores={self.n_jobs} | model={self.esm_model}")

    def run_all(
        self,
        csv_path: str,
        wildtype_csv: str,
        mutation_csv: str,
        clusters_csv: str = "backbone_clusters.csv",
        seq_to_wt_csv: str = "sequence_to_wt.csv",
        cluster_to_wt_csv: str = "cluster_to_wt.csv",
        properties_out: str = "sequences_raw_properties.csv",
        alignment_out: str = "alignment_results.csv",
        kcat_km_out: str = "kcat_km_predictions.csv",
        col_seq: str = "sequence",
    ) -> pd.DataFrame:
        import time
        start_time = time.time()
        df = pd.read_csv(csv_path)

        # ── STEP 0: WT mapping ─────────────────────────────────────────────
        wt_map_exists = (
            os.path.exists(seq_to_wt_csv) and os.path.exists(cluster_to_wt_csv)
        )
        if self.rebuild_wt_map or not wt_map_exists:
            print("\n" + "=" * 60)
            print("STEP 0: Building WT mapping per cluster")
            print("=" * 60)
            build_wt_mapping(
                test_csv=csv_path,
                wt_csv=wildtype_csv,
                clusters_csv=clusters_csv,
                out_cluster_map=cluster_to_wt_csv,
                out_seq_map=seq_to_wt_csv,
                seq_col=col_seq,
            )
        else:
            print(f"\nSTEP 0: WT mapping files exist → skipping "
                  f"(use rebuild_wt_map=True to force rebuild)")

        # ── STEP 1: Properties ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Computing sequence properties")
        print("=" * 60)
        df_props = compute_properties(
            df,
            clusters_csv=clusters_csv,
            seq_to_wt_csv=seq_to_wt_csv,
            wt_csv=wildtype_csv,
            out_path=properties_out,
            n_jobs=self.n_jobs,
            col_seq=col_seq,
            esm_model=self.esm_model,
            esm_cache_dir=self.esm_cache_dir,
        )

        # ── STEP 2: BLOSUM alignment ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 2: Running BLOSUM alignment")
        print("=" * 60)
        alignment_blossum(wildtype_csv, mutation_csv, alignment_out, n_jobs=self.n_jobs)

        # ── STEP 3+4: pH + kcat/km ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 3 & 4: pH optimum + kinetic predictions (parallel)")
        print("=" * 60)
        with ThreadPoolExecutor(max_workers=2) as executor:
            ph_future   = executor.submit(compute_ph_optimum_pred, csv_path)
            kcat_future = executor.submit(compute_kcat_km, df_props, col_seq, kcat_km_out)
            ph_future.result()
            kcat_future.result()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETE! Total time: {elapsed:.2f} seconds")
        print("=" * 60)

        return df_props


# ==============================================================================
# USAGE
# ==============================================================================

if __name__ == "__main__":
    # Requires ESM Forge API key:
    #   export ESM_API_KEY="your_forge_token"
    # Get one at: https://forge.evolutionaryscale.ai

    pipeline = ParallelPipeline(
        n_jobs=-1,
        esm_model="esmc-6b-2024-12",
        esm_cache_dir="cache_forge_esmc",
        rebuild_wt_map=True,  # set True if your sequences or clusters changed
    )
    PATH = '/Users/macbookpro/Documents/ALIGN/code/align-mada-protein/'

    results = pipeline.run_all(
        csv_path=PATH + "dataset/external_petase_expression.csv",
        wildtype_csv=PATH + "dataset/capetase-wildtype.csv",
        mutation_csv=PATH + "dataset/external_petase_expression.csv",
        clusters_csv=PATH + "output/backbone_clusters.csv",
        # seq_to_wt_csv="sequence_to_wt.csv",
        # cluster_to_wt_csv="cluster_to_wt.csv",
        col_seq="seq_aa",

    )