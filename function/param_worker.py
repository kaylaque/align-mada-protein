# scoring_cluster_parallel.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

from seq_features import all_params, net_charge, mean_hydrophobicity, pro_gly_fraction
from esm_cache import esm_ll_for_sequences
from motif_agg_features import (
    find_gxsxg,
    window_charge,
    max_hydrophobic_window,
    max_hydrophobic_run,
)
from alignment import main_process as blossum_align
from ph_optimum import main_process as phOpt
from km_kcat_pred import main_process as kinetic_pred


def zscore(x: np.ndarray) -> np.ndarray:
    """Compute z-score normalization."""
    mu = x.mean()
    sd = x.std() if x.std() > 0 else 1.0
    return (x - mu) / sd


def add_clusters(df: pd.DataFrame, clusters_csv: str, reps_csv: str) -> pd.DataFrame:
    """Add cluster information to dataframe."""
    cl = pd.read_csv(clusters_csv)
    reps = pd.read_csv(reps_csv)

    df = df.merge(cl[["sequence", "cluster_id"]], on="sequence", how="left")
    df = df.merge(reps[["cluster_id", "rep_sequence"]], on="cluster_id", how="left")

    # Handle missing clusters
    missing = df["cluster_id"].isna()
    if missing.any():
        df.loc[missing, "cluster_id"] = -1
        df.loc[missing, "rep_sequence"] = df.loc[missing, "sequence"]

    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def compute_motif_features_batch(sequences: List[str], start_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute motif and local features for a batch of sequences."""
    n = len(sequences)
    motif_present = np.zeros(n, dtype=np.int32)
    motif_center = np.full(n, -1, dtype=np.int32)
    local_c55 = np.zeros(n, dtype=np.float32)
    local_c90 = np.zeros(n, dtype=np.float32)
    
    LOCAL_WIN = 41
    
    for i, seq in enumerate(sequences):
        mp, mc = find_gxsxg(seq)
        motif_present[i] = mp
        motif_center[i] = mc
        
        local_c55[i] = window_charge(seq, mc, LOCAL_WIN, net_charge, 5.5)
        local_c90[i] = window_charge(seq, mc, LOCAL_WIN, net_charge, 9.0)
    
    return motif_present, motif_center, local_c55, local_c90


def compute_aggregation_features_batch(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute aggregation features for a batch of sequences."""
    agg_win = np.array([max_hydrophobic_window(s, win=9) for s in sequences], dtype=np.float32)
    agg_run = np.array([max_hydrophobic_run(s) for s in sequences], dtype=np.float32)
    return agg_win, agg_run


def compute_properties(
    df: pd.DataFrame,
    clusters_csv: str = "backbone_clusters.csv",
    reps_csv: str = "cluster_representatives.csv",
    out_path: str = 'sequences_raw_properties.csv',
    n_jobs: int = -1,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Compute sequence properties with parallel processing.
    
    Args:
        df: Input dataframe with 'sequence' column
        clusters_csv: Path to clusters file
        reps_csv: Path to representatives file
        out_path: Output CSV path
        n_jobs: Number of parallel jobs (-1 for all cores)
        batch_size: Batch size for parallel processing
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Computing properties with {n_jobs} cores...")
    
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["seq_len"] = df["sequence"].str.len()

    # Add cluster information
    df = add_clusters(df, clusters_csv, reps_csv)

    seqs = df["sequence"].tolist()
    reps = df["rep_sequence"].tolist()

    # === ESM scores (these are typically cached, so run serially) ===
    print("Computing ESM scores...")
    esm_seq = esm_ll_for_sequences(seqs)
    esm_rep = esm_ll_for_sequences(reps)

    df["esm_ll"] = esm_seq
    df["esm_ll_rep"] = esm_rep
    df["delta_esm"] = df["esm_ll"] - df["esm_ll_rep"]

    # === Biophysical features (parallel) ===
    print("Computing biophysical features...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        protein_params = list(tqdm(
            executor.map(all_params, seqs),
            total=len(seqs),
            desc="Biophysical features"
        ))
    
    protein_params_df = pd.DataFrame(protein_params)
    df = pd.concat([df, protein_params_df], axis=1)

    # === Motif & local features (parallel batching) ===
    print("Computing motif and local features...")
    n_seqs = len(seqs)
    batches = [(seqs[i:i+batch_size], i) for i in range(0, n_seqs, batch_size)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(compute_motif_features_batch, batch, start) 
                   for batch, start in batches]
        
        motif_results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Motif features"):
            motif_results.append(future.result())
    
    # Combine results
    motif_present = np.concatenate([r[0] for r in motif_results])
    motif_center = np.concatenate([r[1] for r in motif_results])
    local_c55 = np.concatenate([r[2] for r in motif_results])
    local_c90 = np.concatenate([r[3] for r in motif_results])

    df["motif_present"] = motif_present
    df["motif_center"] = motif_center
    df["local_charge_55"] = local_c55
    df["local_charge_90"] = local_c90

    # === Aggregation features (parallel batching) ===
    print("Computing aggregation features...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(compute_aggregation_features_batch, seqs[i:i+batch_size]) 
                   for i in range(0, n_seqs, batch_size)]
        
        agg_results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregation features"):
            agg_results.append(future.result())
    
    agg_win = np.concatenate([r[0] for r in agg_results])
    agg_run = np.concatenate([r[1] for r in agg_results])

    df["agg_win9"] = agg_win
    df["agg_run"] = agg_run

    # === Length penalty ===
    mode_len = int(df["seq_len"].mode().iloc[0])
    len_pen = -np.abs(df["seq_len"].values - mode_len) / mode_len
    df["len_pen"] = len_pen
    
    mp = df["motif_present"].values.astype(np.float32)
    penalty = 1.0
    df['motif_penalty'] = (penalty - mp)

    # === Normalization ===
    print("Normalizing features...")
    col_to_normalize = [
        'delta_esm', 'esm_ll', 'len_pen', 'local_charge_55', 'local_charge_90',
        'agg_win9', 'agg_run', 'IEP', 'charge7', 'charge9', 'charge55',
        'molecular_weight', 'instability_index', 'hydrophobicity', 'pg_fraction'
    ]
    col_normalized = ['z_' + col for col in col_to_normalize]
    
    df[col_normalized] = df[col_to_normalize].apply(zscore)
    
    print(f"Saving to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Properties computation complete!")
    
    return df


def alignment_blossum(wildtype: str, mutation: str, output: str, n_jobs: int = -1):
    """
    Run BLOSUM alignment with full core usage.
    
    Args:
        wildtype: Path to wildtype CSV
        mutation: Path to mutation CSV
        output: Output path
        n_jobs: Number of jobs (-1 for all cores)
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Running BLOSUM alignment with {n_jobs} cores...")
    backend = 'loky'
    blossum_align(wildtype, mutation, n_jobs, backend, output)
    print("BLOSUM alignment complete!")


def compute_ph_optimum_pred(input_csv: str):
    """Compute pH optimum predictions for wildtype and mutations."""
    task = 'pHopt'
    print(f"Computing pH optimum for {input_csv}...")
    output = input_csv.split('.csv')[0] + 'pH_opt.csv'
    phOpt(input_csv, output, task)
    print(f"Computing pH optimum saved in {output}...")


def compute_kcat_km(df: pd.DataFrame, col_seq: str, out_path: str):
    """Compute kcat/km predictions."""
    print(f"Computing kcat/km predictions from column '{col_seq}'...")
    sequences = df[col_seq].tolist()
    kinetic_pred(sequences, out_path)
    print("Kinetic predictions complete!")


class ParallelPipeline:
    """
    Orchestrate the entire pipeline with parallel execution.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        print(f"Initialized pipeline with {self.n_jobs} cores")
    
    def run_all(
        self,
        csv_path: str,
        wildtype_csv: str,
        mutation_csv: str,
        clusters_csv: str = "backbone_clusters.csv",
        reps_csv: str = "cluster_representatives.csv",
        properties_out: str = 'sequences_raw_properties.csv',
        alignment_out: str = 'alignment_results.csv',
        kcat_km_out: str = 'kcat_km_predictions.csv',
        col_seq: str = 'sequence',
    ) -> pd.DataFrame:
        """
        Run the complete pipeline with intelligent parallelization.
        
        BLOSUM alignment uses all cores since it's compute-intensive.
        Other tasks share remaining resources.
        """
        import time
        start_time = time.time()
        df = pd.read_csv(csv_path)
        # Step 1: Compute properties (parallel within)
        print("\n" + "="*60)
        print("STEP 1: Computing sequence properties")
        print("="*60)
        df_props = compute_properties(
            df, clusters_csv, reps_csv, properties_out, 
            n_jobs=self.n_jobs
        )
        
        # Step 2: BLOSUM alignment (uses all cores)
        print("\n" + "="*60)
        print("STEP 2: Running BLOSUM alignment")
        print("="*60)
        alignment_blossum(wildtype_csv, mutation_csv, alignment_out, n_jobs=self.n_jobs)
        
        # Step 3 & 4: Run pH optimum and kcat/km in parallel
        print("\n" + "="*60)
        print("STEP 3 & 4: Running pH optimum and kinetic predictions in parallel")
        print("="*60)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            ph_future = executor.submit(compute_ph_optimum_pred, csv)
            kcat_future = executor.submit(compute_kcat_km, df_props, col_seq, kcat_km_out)
            
            # Wait for completion
            ph_future.result()
            kcat_future.result()
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE! Total time: {elapsed:.2f} seconds")
        print("="*60)
        
        return df_props


# Usage example
if __name__ == "__main__":
    # Example usage
    pipeline = ParallelPipeline(n_jobs=-1)  # Use all cores
    
    results = pipeline.run_all(
        csv_path='sequence.csv',
        wildtype_csv="wildtype.csv",
        mutation_csv="mutations.csv",
        clusters_csv="backbone_clusters.csv",
        reps_csv="cluster_representatives.csv",
    )