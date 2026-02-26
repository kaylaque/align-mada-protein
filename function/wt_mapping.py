# wt_mapping.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def _hash_kmer(kmer: str, dim: int) -> int:
    # hash stabil lintas run (bukan built-in hash python)
    h = 2166136261
    for ch in kmer:
        h ^= ord(ch)
        h *= 16777619
        h &= 0xFFFFFFFF
    return int(h % dim)

def kmer_vec(seq: str, k: int = 3, dim: int = 4096) -> np.ndarray:
    """Hashed k-mer count vector (L2-normalized)."""
    seq = str(seq).strip()
    v = np.zeros(dim, dtype=np.float32)
    if len(seq) < k:
        return v
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        # skip weird chars
        if any(ch not in AA_ALPHABET for ch in km):
            continue
        v[_hash_kmer(km, dim)] += 1.0
    # normalize
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # both are normalized, but keep safe
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@dataclass
class WTRecord:
    wt_id: int
    wt_seq: str
    cds: Optional[str] = None
    wt_len: int = 0

def load_wt_table(wt_csv: str) -> List[WTRecord]:
    wt = pd.read_csv(wt_csv)
    if "Wt AA Sequence" not in wt.columns:
        raise ValueError("WT CSV must contain column 'Wt AA Sequence'")
    cds_col = "CDS" if "CDS" in wt.columns else None

    recs: List[WTRecord] = []
    for i, row in wt.iterrows():
        s = str(row["Wt AA Sequence"]).strip()
        cds = str(row[cds_col]).strip() if cds_col else None
        recs.append(WTRecord(wt_id=int(i), wt_seq=s, cds=cds, wt_len=len(s)))
    return recs

def load_test_sequences(test_csv: str, seq_col: str = "sequence") -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    if seq_col not in df.columns:
        raise ValueError(f"Test CSV must contain column '{seq_col}'")
    df[seq_col] = df[seq_col].astype(str)
    return df

def load_backbone_clusters(cluster_csv: str) -> pd.DataFrame:
    """
    backbone_clusters.csv expected to contain:
      sequence, cluster_id
    """
    cl = pd.read_csv(cluster_csv)
    if "sequence" not in cl.columns or "cluster_id" not in cl.columns:
        raise ValueError("backbone_clusters.csv must contain columns: sequence, cluster_id")
    cl["sequence"] = cl["sequence"].astype(str)
    return cl

def find_wt_in_clusters(test_df: pd.DataFrame, cl_df: pd.DataFrame, wt_records: List[WTRecord],
                        seq_col: str = "sequence") -> pd.DataFrame:
    """
    Return table of WT sequences that appear in TEST with their cluster_id.
    Output columns: sequence, cluster_id, is_wt, wt_id, wt_len
    """
    wt_set = {r.wt_seq for r in wt_records}

    # Merge without losing the sequence column
    merged = test_df[[seq_col]].merge(
        cl_df[["sequence", "cluster_id"]],
        left_on=seq_col,
        right_on="sequence",
        how="left",
        suffixes=("", "_cl"),
    )

    # If seq_col == "sequence", we now have two columns: "sequence" (from left) and "sequence_cl" (from right)
    if "sequence_cl" in merged.columns:
        merged = merged.drop(columns=["sequence_cl"])

    merged["is_wt"] = merged[seq_col].isin(wt_set)

    wt_in_test = merged[merged["is_wt"]].copy()
    wt_id_map = {r.wt_seq: r.wt_id for r in wt_records}
    wt_in_test["wt_id"] = wt_in_test[seq_col].map(wt_id_map).astype(int)
    wt_in_test["wt_len"] = wt_in_test[seq_col].str.len()

    # Normalize output column name to 'sequence'
    if seq_col != "sequence":
        wt_in_test = wt_in_test.rename(columns={seq_col: "sequence"})
    else:
        wt_in_test = wt_in_test.rename(columns={seq_col: "sequence"})

    return wt_in_test[["sequence", "cluster_id", "is_wt", "wt_id", "wt_len"]]
    

def choose_cluster_wt(cluster_id: int, wt_candidates: pd.DataFrame,
                      k: int = 3, dim: int = 4096) -> Tuple[int, str, float]:
    """
    Given WT candidates (rows) that fall into the same cluster, choose ONE WT origin.
    Tie-break:
      1) choose the most common length among WT candidates (mode)
      2) within that length, choose the WT most "central" w.r.t others using k-mer cosine
    Returns: (wt_id, wt_seq, centrality_score)
    """
    sub = wt_candidates[wt_candidates["cluster_id"] == cluster_id].copy()
    if len(sub) == 0:
        return -1, "", 0.0
    if len(sub) == 1:
        row = sub.iloc[0]
        return int(row["wt_id"]), str(row["sequence"]), 1.0

    # length mode
    mode_len = int(sub["wt_len"].value_counts().idxmax())
    sub = sub[sub["wt_len"] == mode_len].copy()
    if len(sub) == 1:
        row = sub.iloc[0]
        return int(row["wt_id"]), str(row["sequence"]), 1.0

    # centrality via average cosine similarity in k-mer space
    vecs = [kmer_vec(s, k=k, dim=dim) for s in sub["sequence"].tolist()]
    sims = np.zeros(len(vecs), dtype=np.float32)
    for i in range(len(vecs)):
        s = 0.0
        for j in range(len(vecs)):
            if i == j:
                continue
            s += cosine(vecs[i], vecs[j])
        sims[i] = s / (len(vecs) - 1)

    best_i = int(np.argmax(sims))
    best_row = sub.iloc[best_i]
    return int(best_row["wt_id"]), str(best_row["sequence"]), float(sims[best_i])

def build_cluster_to_wt_map(test_csv: str, wt_csv: str, backbone_clusters_csv: str,
                            out_cluster_map_csv: str = "cluster_to_wt.csv",
                            out_sequence_map_csv: str = "sequence_to_wt.csv",
                            seq_col: str = "sequence",
                            k: int = 3, dim: int = 4096) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      1) cluster_to_wt.csv: cluster_id -> wt_id, wt_sequence
      2) sequence_to_wt.csv: sequence -> cluster_id -> wt_id, wt_sequence
    """
    wt_records = load_wt_table(wt_csv)
    test_df = load_test_sequences(test_csv, seq_col=seq_col)
    cl_df = load_backbone_clusters(backbone_clusters_csv)

    wt_in_test = find_wt_in_clusters(test_df, cl_df, wt_records, seq_col=seq_col)

    # all cluster ids present in cluster file
    all_clusters = sorted(cl_df["cluster_id"].dropna().unique().tolist())

    rows = []
    for cid in all_clusters:
        wt_id, wt_seq, central = choose_cluster_wt(int(cid), wt_in_test, k=k, dim=dim)
        rows.append({
            "cluster_id": int(cid),
            "wt_id": int(wt_id),
            "wt_sequence": wt_seq,
            "wt_centrality": float(central),
        })
    cluster_map = pd.DataFrame(rows)

    # attach CDS if available
    wt_id_to_cds = {r.wt_id: r.cds for r in wt_records}
    cluster_map["wt_cds"] = cluster_map["wt_id"].map(wt_id_to_cds)

    # map each test sequence to its cluster and WT
    seq_map = test_df[[seq_col]].merge(cl_df, left_on=seq_col, right_on="sequence", how="left").drop(columns=["sequence"])
    seq_map = seq_map.merge(cluster_map[["cluster_id", "wt_id", "wt_sequence"]], on="cluster_id", how="left")

    cluster_map.to_csv(out_cluster_map_csv, index=False)
    seq_map.rename(columns={seq_col: "sequence"}).to_csv(out_sequence_map_csv, index=False)

    return cluster_map, seq_map