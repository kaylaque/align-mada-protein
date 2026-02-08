#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

from forge_esm_scoring import ForgeESMEmbedder, pool_mean

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--clusters", required=True, help="backbone_clusters.csv with columns sequence,cluster_id")
    ap.add_argument("--seq_to_wt_csv", required=True, help="CSV with columns cluster_id, wt_sequence (may have duplicates)")
    ap.add_argument("--out_csv", default="delta_forge_vs_wt.csv")
    ap.add_argument("--model", default="esmc-6b-2024-12")
    ap.add_argument("--cache_dir", default="cache_forge_esmc")
    ap.add_argument("--wt_csv", default="dataset/pet-2025-wildtype-cds.csv")
    args = ap.parse_args()

    token = os.environ["ESM_API_KEY"]

    df_test = pd.read_csv(args.test_csv)
    cl_df = pd.read_csv(args.clusters)
    map_df = pd.read_csv(args.seq_to_wt_csv)

    # validate columns
    if "sequence" not in df_test.columns:
        raise ValueError(f"test_csv columns: {list(df_test.columns)} (expected 'sequence')")
    if "sequence" not in cl_df.columns or "cluster_id" not in cl_df.columns:
        raise ValueError(f"clusters columns: {list(cl_df.columns)} (expected 'sequence','cluster_id')")
    if "cluster_id" not in map_df.columns or "wt_sequence" not in map_df.columns:
        raise ValueError(f"seq_to_wt_csv columns: {list(map_df.columns)} (expected 'cluster_id','wt_sequence')")

    # 1) test -> cluster_id
    merged = df_test.merge(cl_df[["sequence", "cluster_id"]], on="sequence", how="left")
    if merged["cluster_id"].isna().any():
        n = int(merged["cluster_id"].isna().sum())
        raise ValueError(f"{n} test sequences missing cluster_id. Check clusters file matches test CSV.")

    # 2) cluster_id -> wt_sequence (dedupe to avoid cartesian explosion)
    map_unique = (
        map_df[["cluster_id", "wt_sequence"]]
        .dropna()
        .drop_duplicates(subset=["cluster_id"])
        .copy()
    )

    merged = merged.merge(map_unique, on="cluster_id", how="left")

    missing = merged["wt_sequence"].isna() | (merged["wt_sequence"].astype(str).str.strip() == "")
    if missing.any():
        wt_df = pd.read_csv(args.wt_csv)

        # build fallback WT per length (pick first WT for each length)
        wt_df["wt_len"] = wt_df["Wt AA Sequence"].astype(str).str.len()
        wt_by_len = (
            wt_df.sort_values("wt_len")
            .groupby("wt_len")["Wt AA Sequence"]
            .first()
            .to_dict()
        )

        # if some lengths not found, fallback to global first WT
        wt_global = str(wt_df["Wt AA Sequence"].iloc[0])

        # we need seq_len for fallback; get it from clusters file if not already present
        if "seq_len" not in merged.columns:
            # derive from sequence length
            merged["seq_len"] = merged["sequence"].astype(str).str.len()

        def fill_wt(row):
            L = int(row["seq_len"])
            return str(wt_by_len.get(L, wt_global))

        merged.loc[missing, "wt_sequence"] = merged.loc[missing].apply(fill_wt, axis=1)

        print(f"WARNING: filled {int(missing.sum())} sequences with fallback WT by seq_len.")

    print("Rows after merges:", len(merged))
    print("Unique clusters:", merged["cluster_id"].nunique())
    print("Unique WT used:", merged["wt_sequence"].nunique())

    embedder = ForgeESMEmbedder(model=args.model, token=token, cache_dir=args.cache_dir)

    # compute WT pooled vectors (unique WT only)
    wt_unique = merged["wt_sequence"].astype(str).unique().tolist()
    wt_vecs = {}
    for wt in wt_unique:
        emb = embedder.embed_one(wt)
        wt_vecs[wt] = pool_mean(emb)

    # compute pooled vec for each seq and cosine vs WT
    wt_cos = []
    delta_emb = []
    for s, wt in zip(merged["sequence"].astype(str), merged["wt_sequence"].astype(str)):
        emb = embedder.embed_one(s)
        vec = pool_mean(emb)
        sim = cosine_sim(vec, wt_vecs[wt])
        wt_cos.append(sim)
        delta_emb.append(1.0 - sim)

    merged["wt_cosine_sim"] = wt_cos
    merged["delta_emb"] = delta_emb

    merged[["sequence", "wt_sequence", "wt_cosine_sim", "delta_emb"]].to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()