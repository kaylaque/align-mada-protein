# scoring_cluster.py (V2 aggressive)
import numpy as np
import pandas as pd

from function.seq_features import net_charge, mean_hydrophobicity, pro_gly_fraction
from function.esm_cache import esm_ll_for_sequences

from function.motif_agg_features import (
    find_gxsxg,
    window_charge,
    max_hydrophobic_window,
    max_hydrophobic_run,
)

def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median/MAD (lebih stabil untuk outlier)."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)

def add_clusters(df: pd.DataFrame, clusters_csv: str, reps_csv: str) -> pd.DataFrame:
    cl = pd.read_csv(clusters_csv)  # expected: sequence, seq_len, cluster_id (or at least sequence, cluster_id)
    reps = pd.read_csv(reps_csv)    # expected: cluster_id, rep_sequence, ...

    if "sequence" not in cl.columns or "cluster_id" not in cl.columns:
        raise ValueError(f"clusters_csv columns: {list(cl.columns)} expected at least ['sequence','cluster_id']")
    if "cluster_id" not in reps.columns or "rep_sequence" not in reps.columns:
        raise ValueError(f"reps_csv columns: {list(reps.columns)} expected ['cluster_id','rep_sequence']")

    df = df.merge(cl[["sequence", "cluster_id"]], on="sequence", how="left")
    df = df.merge(reps[["cluster_id", "rep_sequence"]], on="cluster_id", how="left")

    # safety: kalau ada missing, jadikan cluster sendiri
    missing = df["cluster_id"].isna() | df["rep_sequence"].isna()
    if missing.any():
        df.loc[missing, "cluster_id"] = -1
        df.loc[missing, "rep_sequence"] = df.loc[missing, "sequence"]

    df["cluster_id"] = df["cluster_id"].astype(int)
    return df

def _cluster_centered(df: pd.DataFrame, col: str) -> np.ndarray:
    """x - mean(x within cluster) (cluster-relative feature)."""
    x = df[col].astype(np.float32)
    mu = df.groupby("cluster_id")[col].transform("mean").astype(np.float32)
    return (x - mu).to_numpy()

def compute_property_scores_clusteraware_v2(
    df: pd.DataFrame,
    clusters_csv: str = "output/param/backbone_clusters.csv",
    reps_csv: str = "output/param/cluster_representatives.csv",
    delta_forge_csv: str | None = None,   # expects columns: sequence, delta_emb OR wt_cosine_sim/delta_emb
) -> pd.DataFrame:
    """
    Aggressive ranking V2:
    - delta_esm backbone-relative (LL(seq)-LL(rep)) + global esm_ll
    - optional delta_emb from Forge cosine vs WT (1-cosine)
    - cluster-centering + robust scaling
    - motif gating keras
    - length penalty non-linear
    - expression: aggregation penalty lebih keras
    """
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

    df["esm_ll"] = np.asarray(esm_seq, dtype=np.float32)
    df["esm_ll_rep"] = np.asarray(esm_rep, dtype=np.float32)
    df["delta_esm"] = df["esm_ll"] - df["esm_ll_rep"]

    # === optional: Forge delta_emb ===
    if delta_forge_csv:
        dfg = pd.read_csv(delta_forge_csv)
        if "sequence" not in dfg.columns:
            raise ValueError(f"delta_forge_csv columns: {list(dfg.columns)} expected 'sequence'")
        if "delta_emb" not in dfg.columns:
            raise ValueError(f"delta_forge_csv columns: {list(dfg.columns)} expected 'delta_emb'")
        dfg["sequence"] = dfg["sequence"].astype(str).str.strip()
        dfg = dfg[["sequence", "delta_emb"]].drop_duplicates("sequence")
        df = df.merge(dfg, on="sequence", how="left")
        # kalau ada missing, isi dengan median (aman untuk robust_z)
        if df["delta_emb"].isna().any():
            df["delta_emb"] = df["delta_emb"].fillna(df["delta_emb"].median())
    else:
        df["delta_emb"] = np.nan

    # === biophysical features ===
    charge_55 = np.array([net_charge(s, 5.5) for s in seqs], dtype=np.float32)
    charge_90 = np.array([net_charge(s, 9.0) for s in seqs], dtype=np.float32)
    hydro     = np.array([mean_hydrophobicity(s) for s in seqs], dtype=np.float32)
    pro_gly   = np.array([pro_gly_fraction(s) for s in seqs], dtype=np.float32)

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

    LOCAL_WIN = 41  # +/-20
    local_c55 = np.array([
        window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 5.5)
        for i in range(len(seqs))
    ], dtype=np.float32)
    local_c90 = np.array([
        window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 9.0)
        for i in range(len(seqs))
    ], dtype=np.float32)

    df["local_charge_55"] = local_c55
    df["local_charge_90"] = local_c90

    # aggregation proxies (expression)
    agg_win = np.array([max_hydrophobic_window(s, win=9) for s in seqs], dtype=np.float32)
    agg_run = np.array([max_hydrophobic_run(s) for s in seqs], dtype=np.float32)
    df["agg_win9"] = agg_win
    df["agg_run"]  = agg_run

    # === length penalty (non-linear, lebih “tajam” untuk truncations) ===
    mode_len = int(df["seq_len"].mode().iloc[0])  # mostly 257/259
    len_dev = np.abs(df["seq_len"].values.astype(np.float32) - mode_len) / float(mode_len)
    len_pen = -(len_dev ** 1.5)   # exponent => truncations dihukum lebih keras
    df["len_pen"] = len_pen

    # === cluster-centered features (ranking intra-backbone) ===
    desm_c = _cluster_centered(df, "delta_esm")
    esm_c  = _cluster_centered(df, "esm_ll")
    lenp_c = _cluster_centered(df, "len_pen")
    hydro_c = _cluster_centered(df, "hydro")
    pg_c    = _cluster_centered(df, "pro_gly")
    lc55_c  = _cluster_centered(df, "local_charge_55")
    lc90_c  = _cluster_centered(df, "local_charge_90")
    aggw_c  = _cluster_centered(df, "agg_win9")
    aggr_c  = _cluster_centered(df, "agg_run")

    if np.isfinite(df["delta_emb"].to_numpy()).all():
        demb_c = _cluster_centered(df, "delta_emb")
        z_demb_c = robust_z(demb_c)
        z_demb_g = robust_z(df["delta_emb"].to_numpy(dtype=np.float32))
    else:
        z_demb_c = None
        z_demb_g = None

    # === robust scaling (aggressive ranking friendly) ===
    z_desm_c = robust_z(desm_c)
    z_esm_g  = robust_z(df["esm_ll"].to_numpy(dtype=np.float32))
    z_esm_c  = robust_z(esm_c)

    z_hydro_c = robust_z(hydro_c)
    z_pg_c    = robust_z(pg_c)
    z_lenp_c  = robust_z(lenp_c)

    z_lc55_c  = robust_z(lc55_c)
    z_lc90_c  = robust_z(lc90_c)

    z_aggw_c  = robust_z(aggw_c)
    z_aggr_c  = robust_z(aggr_c)

    # motif (binary gate)
    mp = df["motif_present"].values.astype(np.float32)
    motif_pen = (1.0 - mp)  # 1 if missing

    # === SCORING V2 (AGGRESSIVE) ===
    # activity core: dorong separation keras, fokus intra-cluster + sedikit global
    # delta_emb: makin besar = makin jauh dari WT -> biasanya kurang stabil/fitness => NEGATIVE
    demb_term = 0.0
    if z_demb_c is not None:
        demb_term = -0.9 * z_demb_c - 0.2 * z_demb_g  # aggressive penalty for far-from-WT

    act_core = (
        1.7 * z_desm_c +
        0.35 * z_esm_c +
        0.20 * z_esm_g +
        0.55 * z_hydro_c +
        0.60 * z_lenp_c -
        0.75 * np.abs(z_lc55_c) +
        demb_term
    )

    # motif gating keras (aktivitas drop kuat jika motif hilang)
    act1 = act_core - 3.2 * motif_pen
    act2 = (
        act_core +
        0.15 * z_pg_c -
        0.55 * np.abs(z_lc90_c) -
        3.0 * motif_pen
    )

    # expression: aggregation penalty lebih keras + demb_term juga relevan
    expr = (
        1.35 * z_desm_c +
        0.25 * z_esm_c +
        0.20 * z_esm_g -
        0.85 * z_hydro_c +
        0.55 * z_pg_c +
        0.65 * z_lenp_c -
        1.05 * z_aggw_c -
        0.95 * z_aggr_c +
        0.40 * (demb_term if isinstance(demb_term, np.ndarray) else 0.0)
    )

    df["activity_1_pred"] = act1.astype(np.float32)
    df["activity_2_pred"] = act2.astype(np.float32)
    df["expression_pred"] = expr.astype(np.float32)

    # guard: no NaN/inf
    for c in ["activity_1_pred", "activity_2_pred", "expression_pred"]:
        df[c] = np.nan_to_num(df[c].to_numpy(dtype=np.float32), nan=-10.0, posinf=10.0, neginf=-10.0)

    return df

# Backward-compatible alias (kalau script lain masih import nama lama)
def compute_property_scores_clusteraware(df: pd.DataFrame, clusters_csv="output/param/backbone_clusters.csv", reps_csv="output/param/cluster_representatives.csv"):
    return compute_property_scores_clusteraware_v2(df, clusters_csv=clusters_csv, reps_csv=reps_csv, delta_forge_csv=None)


#============================================================
# # scoring_cluster.py
# import numpy as np
# import pandas as pd

# from seq_features import net_charge, mean_hydrophobicity, pro_gly_fraction
# from esm_cache import esm_ll_for_sequences

# from motif_agg_features import (
#     find_gxsxg,
#     window_charge,
#     max_hydrophobic_window,
#     max_hydrophobic_run,
# )

# def zscore(x: np.ndarray) -> np.ndarray:
#     mu = x.mean()
#     sd = x.std() if x.std() > 0 else 1.0
#     return (x - mu) / sd

# def add_clusters(df: pd.DataFrame, clusters_csv: str, reps_csv: str) -> pd.DataFrame:
#     cl = pd.read_csv(clusters_csv)  # sequence, seq_len, cluster_id
#     reps = pd.read_csv(reps_csv)    # cluster_id, rep_sequence, ...

#     # map sequence -> cluster_id
#     df = df.merge(cl[["sequence", "cluster_id"]], on="sequence", how="left")

#     # map cluster_id -> rep_sequence
#     df = df.merge(reps[["cluster_id", "rep_sequence"]], on="cluster_id", how="left")

#     if df["cluster_id"].isna().any():
#         # jika ada yang tidak kebaca, set cluster_id unik untuk safety
#         missing = df["cluster_id"].isna()
#         df.loc[missing, "cluster_id"] = -1
#         df.loc[missing, "rep_sequence"] = df.loc[missing, "sequence"]

#     df["cluster_id"] = df["cluster_id"].astype(int)
#     return df

# def compute_property_scores_clusteraware(
#     df: pd.DataFrame,
#     clusters_csv: str = "backbone_clusters.csv",
#     reps_csv: str = "cluster_representatives.csv",
# ) -> pd.DataFrame:

#     df = df.copy()
#     df["sequence"] = df["sequence"].astype(str).str.strip()
#     df["seq_len"] = df["sequence"].str.len()

#     # === attach cluster_id and rep_sequence ===
#     df = add_clusters(df, clusters_csv, reps_csv)

#     seqs = df["sequence"].tolist()
#     reps = df["rep_sequence"].tolist()

#     # === ESM scores with caching ===
#     esm_seq = esm_ll_for_sequences(seqs)
#     esm_rep = esm_ll_for_sequences(reps)

#     df["esm_ll"] = esm_seq
#     df["esm_ll_rep"] = esm_rep
#     df["delta_esm"] = df["esm_ll"] - df["esm_ll_rep"]

#     # === biophysical features ===
#     charge_55 = np.array([net_charge(s, 5.5) for s in seqs])
#     charge_90 = np.array([net_charge(s, 9.0) for s in seqs])
#     hydro     = np.array([mean_hydrophobicity(s) for s in seqs])
#     pro_gly   = np.array([pro_gly_fraction(s) for s in seqs])

#     df["charge_55"] = charge_55
#     df["charge_90"] = charge_90
#     df["hydro"] = hydro
#     df["pro_gly"] = pro_gly

#     # === motif & local features ===
#     motif_present = np.zeros(len(seqs), dtype=np.int32)
#     motif_center  = np.full(len(seqs), -1, dtype=np.int32)

#     for i, s in enumerate(seqs):
#         mp, mc = find_gxsxg(s)
#         motif_present[i] = mp
#         motif_center[i] = mc

#     df["motif_present"] = motif_present
#     df["motif_center"] = motif_center

#     # local charge around motif (window 41 aa -> +/-20)
#     LOCAL_WIN = 41
#     local_c55 = np.array([
#         window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 5.5)
#         for i in range(len(seqs))
#     ])
#     local_c90 = np.array([
#         window_charge(seqs[i], motif_center[i], LOCAL_WIN, net_charge, 9.0)
#         for i in range(len(seqs))
#     ])

#     df["local_charge_55"] = local_c55
#     df["local_charge_90"] = local_c90

#     # aggregation proxies (for expression)
#     agg_win = np.array([max_hydrophobic_window(s, win=9) for s in seqs], dtype=np.float32)
#     agg_run = np.array([max_hydrophobic_run(s) for s in seqs], dtype=np.float32)

#     df["agg_win9"] = agg_win
#     df["agg_run"]  = agg_run

#     # === length penalty (important because you have huge length variance) ===
#     mode_len = int(df["seq_len"].mode().iloc[0])  # likely 259
#     len_pen = -np.abs(df["seq_len"].values - mode_len) / mode_len
#     df["len_pen"] = len_pen

#     # === normalize ===
#     z_desm  = zscore(df["delta_esm"].values)
#     z_esm   = zscore(df["esm_ll"].values)   # keep some global signal
#     z_hydro = zscore(hydro)
#     z_pg    = zscore(pro_gly)
#     z_c55   = zscore(charge_55)
#     z_c90   = zscore(charge_90)
#     z_lenp  = zscore(len_pen)
#     z_lc55 = zscore(local_c55)
#     z_lc90 = zscore(local_c90)
#     z_aggw = zscore(agg_win)
#     z_aggr = zscore(agg_run)

#     # motif presence: jangan zscore (biner), tapi dipakai sebagai gate/penalti
#     mp = df["motif_present"].values.astype(np.float32)

#     # === scoring rules ===
#     # Intuition:
#     # - delta_esm is the backbone-relative mutational signal (most important)
#     # - small amount of global esm helps compare across clusters
#     # - length penalty prevents truncations from gaming score
#     # - charges + hydro + flexibility shape pH and expression

#     # Penalti kuat jika motif GXSXG tidak ada
#     # (Kamu bisa tune angka 2.0 -> 3.0 kalau mau lebih keras)
#     motif_pen = (1.0 - mp)  # 1 jika motif hilang, 0 jika ada

#     act1 = (
#         1.4 * z_desm +
#         0.4 * z_esm +
#         0.4 * z_hydro -
#         0.6 * np.abs(z_lc55) +     # local charge lebih relevan daripada global
#         0.5 * z_lenp -
#         2.5 * motif_pen            # drop activity kalau motif hilang
#     )

#     act2 = (
#         1.4 * z_desm +
#         0.4 * z_esm +
#         0.6 * z_hydro -
#         0.4 * np.abs(z_lc90) +
#         0.3 * z_pg +
#         0.5 * z_lenp -
#         2.3 * motif_pen
#     )

#     # Expression: penalti agregasi (hydrophobic window + hydrophobic run)
#     expr = (
#         1.2 * z_desm +
#         0.4 * z_esm -
#         0.7 * z_hydro +
#         0.5 * z_pg +
#         0.6 * z_lenp -
#         0.8 * z_aggw -
#         0.6 * z_aggr
#     )

#     df["activity_1_pred"] = act1
#     df["activity_2_pred"] = act2
#     df["expression_pred"] = expr

#     return df