#!/usr/bin/env python3
"""
Cluster backbone sequences using k-mer hashing + MiniBatchKMeans (CPU-friendly).

Input : predictive-pet-zero-shot-test-2025.csv  (must have 'sequence' column)
Output:
  - backbone_clusters.csv      : sequence -> cluster_id (global)
  - cluster_representatives.csv: per cluster representative sequence + stats
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


def kmer_hash_vector(seq: str, k: int = 3, dim: int = 2**16) -> np.ndarray:
    """
    Convert sequence to a fixed-size hashed k-mer count vector (dense).
    dim=65536 is a good compromise for speed/quality.
    """
    v = np.zeros(dim, dtype=np.float32)
    seq = str(seq).strip()
    if len(seq) < k:
        return v

    # rolling k-mers
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        # Python hash is salted per process; use stable hash:
        h = (hash(kmer) & 0xFFFFFFFF) % dim
        v[h] += 1.0

    # length normalization helps fair comparison across lengths
    total = v.sum()
    if total > 0:
        v /= total
    return v


def build_matrix(seqs, k=3, dim=2**16) -> np.ndarray:
    X = np.stack([kmer_hash_vector(s, k=k, dim=dim) for s in seqs], axis=0)
    # L2 normalize for cosine-like behavior
    X = normalize(X, norm="l2", axis=1)
    return X


def choose_k(n: int) -> int:
    """
    Heuristic number of clusters.
    You can tune this later.
    """
    if n < 50:
        return 2
    if n < 200:
        return 3
    if n < 600:
        return 5
    if n < 1500:
        return 8
    return 12


def cluster_group(seqs, random_state=42, kmer_k=3, dim=2**16):
    X = build_matrix(seqs, k=kmer_k, dim=dim)
    n = X.shape[0]
    n_clusters = choose_k(n)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=256,
        n_init="auto",
        max_iter=200,
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_  # already in feature space
    centers = normalize(centers, norm="l2", axis=1)

    # representative per cluster = closest to centroid (max cosine similarity)
    reps = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sims = (X[idx] @ centers[c].reshape(-1, 1)).ravel()
        best_local = idx[int(np.argmax(sims))]
        reps[c] = {
            "rep_index": int(best_local),
            "rep_sequence": seqs[best_local],
            "cluster_size": int(len(idx)),
            "avg_sim_to_center": float(np.mean(sims)),
        }

    return labels, reps


def main(input_csv: str, out_clusters: str, out_reps: str, kmer_k: int):
    df = pd.read_csv(input_csv)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    # Basic clean
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df = df[df["sequence"].str.len() > 0].copy()

    # Group by length first (very important for your dataset)
    df["seq_len"] = df["sequence"].str.len()
    length_counts = df["seq_len"].value_counts().to_dict()

    # We'll cluster only large groups; small groups get their own clusters.
    # You can tune threshold.
    BIG_GROUP_THRESHOLD = 50

    global_cluster_id = 0
    cluster_ids = np.full(len(df), -1, dtype=int)
    rep_rows = []

    for L, sub_idx in df.groupby("seq_len").groups.items():
        sub_idx = list(sub_idx)
        n = len(sub_idx)

        seqs = df.loc[sub_idx, "sequence"].tolist()

        if n < BIG_GROUP_THRESHOLD:
            # Each sequence gets its own cluster (or put all small ones into one bucket)
            for j, row_i in enumerate(sub_idx):
                cluster_ids[row_i] = global_cluster_id
                rep_rows.append({
                    "cluster_id": global_cluster_id,
                    "seq_len": int(L),
                    "cluster_size": 1,
                    "rep_sequence": df.loc[row_i, "sequence"],
                    "avg_sim_to_center": 1.0,
                })
                global_cluster_id += 1
            continue

        labels, reps = cluster_group(seqs, kmer_k=kmer_k)

        # Assign cluster ids (local -> global)
        local_to_global = {}
        for c in sorted(set(labels)):
            local_to_global[c] = global_cluster_id
            info = reps.get(c, None)
            rep_rows.append({
                "cluster_id": global_cluster_id,
                "seq_len": int(L),
                "cluster_size": int(info["cluster_size"]) if info else int(np.sum(labels == c)),
                "rep_sequence": info["rep_sequence"] if info else "",
                "avg_sim_to_center": info["avg_sim_to_center"] if info else np.nan,
            })
            global_cluster_id += 1

        for i_local, row_i in enumerate(sub_idx):
            cluster_ids[row_i] = local_to_global[int(labels[i_local])]

        print(f"[len={L}] n={n} -> clusters={len(local_to_global)}")

    df_out = df[["sequence", "seq_len"]].copy()
    df_out["cluster_id"] = cluster_ids

    df_out.to_csv(out_clusters, index=False)
    pd.DataFrame(rep_rows).to_csv(out_reps, index=False)

    print("\nDone.")
    print(f"Wrote: {out_clusters}")
    print(f"Wrote: {out_reps}")
    print(f"Total clusters: {global_cluster_id}")
    print("Top lengths:", dict(Counter(df_out["seq_len"]).most_common(5)))
    print("Length counts:", {k: length_counts[k] for k in sorted(length_counts)[:5]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="predictive-pet-zero-shot-test-2025.csv")
    ap.add_argument("--out-clusters", default="backbone_clusters.csv")
    ap.add_argument("--out-reps", default="cluster_representatives.csv")
    ap.add_argument("--kmer-k", type=int, default=3, help="k-mer size (3 recommended)")
    args = ap.parse_args()
    main(args.input, args.out_clusters, args.out_reps, args.kmer_k)
