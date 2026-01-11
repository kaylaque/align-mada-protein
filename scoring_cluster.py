# scoring_cluster.py
import numpy as np
import pandas as pd

from seq_features import net_charge, mean_hydrophobicity, pro_gly_fraction
from esm_cache import esm_ll_for_sequences

from motif_agg_features import (
    find_gxsxg,
    window_charge,
    max_hydrophobic_window,
    max_hydrophobic_run,
)

def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std() if x.std() > 0 else 1.0
    return (x - mu) / sd

def add_clusters(df: pd.DataFrame, clusters_csv: str, reps_csv: str) -> pd.DataFrame:
    cl = pd.read_csv(clusters_csv)  # sequence, seq_len, cluster_id
    reps = pd.read_csv(reps_csv)    # cluster_id, rep_sequence, ...

    # map sequence -> cluster_id
    df = df.merge(cl[["sequence", "cluster_id"]], on="sequence", how="left")

    # map cluster_id -> rep_sequence
    df = df.merge(reps[["cluster_id", "rep_sequence"]], on="cluster_id", how="left")

    if df["cluster_id"].isna().any():
        # jika ada yang tidak kebaca, set cluster_id unik untuk safety
        missing = df["cluster_id"].isna()
        df.loc[missing, "cluster_id"] = -1
        df.loc[missing, "rep_sequence"] = df.loc[missing, "sequence"]

    df["cluster_id"] = df["cluster_id"].astype(int)
    return df

def compute_property_scores_clusteraware(
    df: pd.DataFrame,
    clusters_csv: str = "backbone_clusters.csv",
    reps_csv: str = "cluster_representatives.csv",
) -> pd.DataFrame:

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["seq_len"] = df["sequence"].str.len()

    # === attach cluster_id and rep_sequence ===
    df = add_clusters(df, clusters_csv, reps_csv)

    seqs = df["sequence"].tolist()
    reps = df["rep_sequence"].tolist()

    # === ESM scores with caching ===
    esm_seq = esm_ll_for_sequences(seqs)
    esm_rep = esm_ll_for_sequences(reps)

    df["esm_ll"] = esm_seq
    df["esm_ll_rep"] = esm_rep
    df["delta_esm"] = df["esm_ll"] - df["esm_ll_rep"]

    # === biophysical features ===
    charge_55 = np.array([net_charge(s, 5.5) for s in seqs])
    charge_90 = np.array([net_charge(s, 9.0) for s in seqs])
    hydro     = np.array([mean_hydrophobicity(s) for s in seqs])
    pro_gly   = np.array([pro_gly_fraction(s) for s in seqs])

    df["charge_55"] = charge_55
    df["charge_90"] = charge_90
    df["hydro"] = hydro
    df["pro_gly"] = pro_gly

    # === motif & local features ===
    motif_present = np.zeros(len(seqs), dtype=np.int32)
    motif_center  = np.full(len(seqs), -1, dtype=np.int32)

    for i, s in enumerate(seqs):
        mp, mc = find_gxsxg(s)
        motif_present[i] = mp
        motif_center[i] = mc

    df["motif_present"] = motif_present
    df["motif_center"] = motif_center

    # local charge around motif (window 41 aa -> +/-20)
    LOCAL_WIN = 41
    local_c55 = np.array([
        window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 5.5)
        for i in range(len(seqs))
    ])
    local_c90 = np.array([
        window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 9.0)
        for i in range(len(seqs))
    ])

    df["local_charge_55"] = local_c55
    df["local_charge_90"] = local_c90

    # aggregation proxies (for expression)
    agg_win = np.array([max_hydrophobic_window(s, win=9) for s in seqs], dtype=np.float32)
    agg_run = np.array([max_hydrophobic_run(s) for s in seqs], dtype=np.float32)

    df["agg_win9"] = agg_win
    df["agg_run"]  = agg_run

    # === length penalty (important because you have huge length variance) ===
    mode_len = int(df["seq_len"].mode().iloc[0])  # likely 259
    len_pen = -np.abs(df["seq_len"].values - mode_len) / mode_len
    df["len_pen"] = len_pen

    # === normalize ===
    z_desm  = zscore(df["delta_esm"].values)
    z_esm   = zscore(df["esm_ll"].values)   # keep some global signal
    z_hydro = zscore(hydro)
    z_pg    = zscore(pro_gly)
    z_c55   = zscore(charge_55)
    z_c90   = zscore(charge_90)
    z_lenp  = zscore(len_pen)
    z_lc55 = zscore(local_c55)
    z_lc90 = zscore(local_c90)
    z_aggw = zscore(agg_win)
    z_aggr = zscore(agg_run)

    # motif presence: jangan zscore (biner), tapi dipakai sebagai gate/penalti
    mp = df["motif_present"].values.astype(np.float32)

    # === scoring rules ===
    # Intuition:
    # - delta_esm is the backbone-relative mutational signal (most important)
    # - small amount of global esm helps compare across clusters
    # - length penalty prevents truncations from gaming score
    # - charges + hydro + flexibility shape pH and expression

    # Penalti kuat jika motif GXSXG tidak ada
    # (Kamu bisa tune angka 2.0 -> 3.0 kalau mau lebih keras)
    motif_pen = (1.0 - mp)  # 1 jika motif hilang, 0 jika ada

    act1 = (
        1.4 * z_desm +
        0.4 * z_esm +
        0.4 * z_hydro -
        0.6 * np.abs(z_lc55) +     # local charge lebih relevan daripada global
        0.5 * z_lenp -
        2.5 * motif_pen            # drop activity kalau motif hilang
    )

    act2 = (
        1.4 * z_desm +
        0.4 * z_esm +
        0.6 * z_hydro -
        0.4 * np.abs(z_lc90) +
        0.3 * z_pg +
        0.5 * z_lenp -
        2.3 * motif_pen
    )

    # Expression: penalti agregasi (hydrophobic window + hydrophobic run)
    expr = (
        1.2 * z_desm +
        0.4 * z_esm -
        0.7 * z_hydro +
        0.5 * z_pg +
        0.6 * z_lenp -
        0.8 * z_aggw -
        0.6 * z_aggr
    )

    df["activity_1_pred"] = act1
    df["activity_2_pred"] = act2
    df["expression_pred"] = expr

    return df