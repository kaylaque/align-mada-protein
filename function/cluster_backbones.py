#!/usr/bin/env python3
"""
Cluster backbone sequences using k-mer hashing + MiniBatchKMeans (CPU-friendly).
Memory-optimised version: reduced hash dim, sparse-style batching, gc-aggressive.

Input : any CSV with a sequence column
Output:
  - backbone_clusters.csv       : sequence, seq_len, cluster_id
  - cluster_representatives.csv : cluster_id, seq_len, cluster_size,
                                  rep_sequence, avg_sim_to_center
"""
import gc
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


# ── tuneable constants ────────────────────────────────────────────────────────
# REDUCED from 2**16 (65 536) to 2**12 (4 096)
# 4096 dims × float32 × 10 000 seqs = ~160 MB   (was 2.6 GB)
# Quality loss is minimal for 3-mer protein sequences (only 20^3 = 8000 possible
# 3-mers, so 4096 buckets still captures most signal after hashing collisions).
DEFAULT_DIM = 2 ** 12   # 4 096
DEFAULT_K   = 3
BIG_GROUP_THRESHOLD = 100


# ── k-mer encoding ────────────────────────────────────────────────────────────

def kmer_hash_vector(seq: str, k: int = DEFAULT_K, dim: int = DEFAULT_DIM) -> np.ndarray:
    """
    Fixed-size hashed k-mer frequency vector, length-normalised, float32.
    Uses a stable (non-salted) FNV-1a hash so results are reproducible
    across Python processes (unlike built-in hash()).
    """
    v = np.zeros(dim, dtype=np.float32)
    seq = str(seq).strip()
    if len(seq) < k:
        return v

    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        # FNV-1a 32-bit — stable across runs
        h = 2166136261
        for ch in kmer:
            h ^= ord(ch)
            h  = (h * 16777619) & 0xFFFFFFFF
        v[h % dim] += 1.0

    total = v.sum()
    if total > 0:
        v /= total
    return v


def build_matrix_batched(seqs: list, k: int = DEFAULT_K,
                         dim: int = DEFAULT_DIM,
                         batch: int = 512) -> np.ndarray:
    """
    Build L2-normalised k-mer matrix in small batches to keep peak RAM low.
    Returns float32 array of shape (N, dim).
    """
    N = len(seqs)
    X = np.empty((N, dim), dtype=np.float32)
    for start in range(0, N, batch):
        end = min(start + batch, N)
        for i, s in enumerate(seqs[start:end]):
            X[start + i] = kmer_hash_vector(s, k=k, dim=dim)
    # L2 normalise in-place (avoids a second copy)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    return X


# ── k heuristic ──────────────────────────────────────────────────────────────

def choose_k(n: int) -> int:
    """Heuristic number of KMeans clusters for a length group of size n."""
    if n < 50:   return 2
    if n < 200:  return 3
    if n < 600:  return 5
    if n < 1500: return 8
    return 12


# ── per-length-group clustering ───────────────────────────────────────────────

def cluster_group(seqs: list, random_state: int = 42,
                  kmer_k: int = DEFAULT_K, dim: int = DEFAULT_DIM):
    """
    Cluster one length-group of sequences.
    Returns (labels array, reps dict).
    Matrix is built in batches and freed immediately after fitting.
    """
    X = build_matrix_batched(seqs, k=kmer_k, dim=dim)
    n_clusters = min(choose_k(len(seqs)), len(seqs))

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(256, len(seqs)),
        n_init="auto",
        max_iter=200,
    )
    labels = km.fit_predict(X)

    centers = normalize(km.cluster_centers_.astype(np.float32), norm="l2", axis=1)

    reps = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # dot product = cosine sim (both L2-normalised)
        sims = X[idx] @ centers[c]
        best_local = idx[int(np.argmax(sims))]
        reps[c] = {
            "rep_index":        int(best_local),
            "rep_sequence":     seqs[best_local],
            "cluster_size":     int(len(idx)),
            "avg_sim_to_center": float(np.mean(sims)),
        }

    # free the big matrix immediately
    del X, centers
    gc.collect()

    return labels, reps


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_csv: str, out_clusters: str, out_reps: str,
         kmer_k: int, dim: int, sequence_col: str):

    print(f"Reading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)

    if sequence_col not in df.columns:
        raise ValueError(
            f"Column '{sequence_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[sequence_col] = df[sequence_col].astype(str).str.strip()
    df = df[df[sequence_col].str.len() > 0].reset_index(drop=True)
    df["seq_len"] = df[sequence_col].str.len()

    print(f"Total sequences : {len(df)}")
    print(f"Unique lengths  : {df['seq_len'].nunique()}")
    print(f"k-mer k={kmer_k}, hash dim={dim}")
    print(f"Large-group threshold: {BIG_GROUP_THRESHOLD}")
    print()

    global_cluster_id = 0
    cluster_ids = np.full(len(df), -1, dtype=np.int32)
    rep_rows: list = []

    for L, sub_idx in df.groupby("seq_len").groups.items():
        sub_idx = list(sub_idx)
        n = len(sub_idx)
        seqs = df.loc[sub_idx, sequence_col].tolist()

        if n < BIG_GROUP_THRESHOLD:
            # small group: every sequence is its own cluster
            for row_i, seq in zip(sub_idx, seqs):
                cluster_ids[row_i] = global_cluster_id
                rep_rows.append({
                    "cluster_id":        global_cluster_id,
                    "seq_len":           int(L),
                    "cluster_size":      1,
                    "rep_sequence":      seq,
                    "avg_sim_to_center": 1.0,
                })
                global_cluster_id += 1
            continue

        # large group: MiniBatchKMeans
        labels, reps = cluster_group(seqs, kmer_k=kmer_k, dim=dim)

        local_to_global: dict = {}
        for c in sorted(set(labels)):
            local_to_global[c] = global_cluster_id
            info = reps.get(c)
            rep_rows.append({
                "cluster_id":        global_cluster_id,
                "seq_len":           int(L),
                "cluster_size":      int(info["cluster_size"]) if info else int(np.sum(labels == c)),
                "rep_sequence":      info["rep_sequence"] if info else "",
                "avg_sim_to_center": info["avg_sim_to_center"] if info else float("nan"),
            })
            global_cluster_id += 1

        for i_local, row_i in enumerate(sub_idx):
            cluster_ids[row_i] = local_to_global[int(labels[i_local])]

        print(f"  [len={L:>5}] n={n:>6} → {len(local_to_global)} clusters")

        del seqs, labels, reps, local_to_global
        gc.collect()

    # ── write outputs ─────────────────────────────────────────────────────────
    df_out = df[[sequence_col, "seq_len"]].copy()
    df_out = df_out.rename(columns={sequence_col: "sequence"})
    df_out["cluster_id"] = cluster_ids

    df_out.to_csv(out_clusters, index=False)
    pd.DataFrame(rep_rows).to_csv(out_reps, index=False)

    print()
    print("Done.")
    print(f"  Wrote : {out_clusters}")
    print(f"  Wrote : {out_reps}")
    print(f"  Total clusters : {global_cluster_id}")
    print(f"  Top lengths    : {dict(Counter(df_out['seq_len']).most_common(5))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cluster protein sequences by k-mer hashing + MiniBatchKMeans"
    )
    ap.add_argument("--input",       required=True,
                    help="Input CSV (must contain sequence column)")
    ap.add_argument("--out-clusters", default="backbone_clusters.csv",
                    help="Output path for backbone_clusters.csv")
    ap.add_argument("--out-reps",     default="cluster_representatives.csv",
                    help="Output path for cluster_representatives.csv")
    ap.add_argument("--kmer-k",  type=int, default=DEFAULT_K,
                    help=f"k-mer size (default {DEFAULT_K})")
    ap.add_argument("--dim",     type=int, default=DEFAULT_DIM,
                    help=f"Hash vector dimension (default {DEFAULT_DIM}). "
                         f"Lower = less RAM. 4096 works well for proteins.")
    ap.add_argument("--seq-col", type=str, default="sequence",
                    help="Column name for sequences (default: 'sequence')")
    args = ap.parse_args()

    main(
        input_csv    = args.input,
        out_clusters = args.out_clusters,
        out_reps     = args.out_reps,
        kmer_k       = args.kmer_k,
        dim          = args.dim,
        sequence_col = args.seq_col,
    )