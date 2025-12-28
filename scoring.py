# scoring.py
import numpy as np
import pandas as pd
from esm_scoring import score_sequences_esm
from seq_features import net_charge, mean_hydrophobicity, pro_gly_fraction

def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std() if x.std() > 0 else 1.0
    return (x - mu) / sd

def compute_property_scores(df: pd.DataFrame) -> pd.DataFrame:
    seqs = df["sequence"].tolist()

    # ---- ESM zero-shot fitness ----
    esm_ll = score_sequences_esm(seqs)

    # ---- Biochemical features ----
    charge_55 = np.array([net_charge(s, 5.5) for s in seqs])
    charge_90 = np.array([net_charge(s, 9.0) for s in seqs])
    hydro     = np.array([mean_hydrophobicity(s) for s in seqs])
    pro_gly   = np.array([pro_gly_fraction(s) for s in seqs])

    df["esm_ll"]    = esm_ll
    df["charge_55"] = charge_55
    df["charge_90"] = charge_90
    df["hydro"]     = hydro
    df["pro_gly"]   = pro_gly

    # ---- Normalisasi ----
    z_esm   = zscore(esm_ll)
    z_hydro = zscore(hydro)
    z_pg    = zscore(pro_gly)
    z_c55   = zscore(charge_55)
    z_c90   = zscore(charge_90)

    # === Heuristic scoring ===
    # Activity_1: pH 5.5 (lebih penalti kalau charge ekstrem)
    act1 = (
        1.2 * z_esm +
        0.5 * z_hydro -
        0.7 * np.abs(z_c55)
    )

    # Activity_2: pH 9.0 (basa, fleksibilitas sedikit menguntungkan)
    act2 = (
        1.2 * z_esm +
        0.7 * z_hydro -
        0.3 * np.abs(z_c90) +
        0.4 * z_pg
    )

    # Expression: stabil tapi tidak terlalu hydrophobic, sedikit fleksibel
    expr = (
        1.0 * z_esm -
        0.8 * z_hydro +
        0.5 * z_pg
    )

    df["activity_1_pred"] = act1
    df["activity_2_pred"] = act2
    df["expression_pred"] = expr

    return df
