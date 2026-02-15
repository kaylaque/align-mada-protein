#!/usr/bin/env python3
'''
python kinetics_no_ode.py \
  --input pred_kcat_km.csv \
  --output v0_out.csv \
  --PET0 1.0 \
  --E_mg_per_mL 1.0 \
  --MW_kDa 30 \
  --volume_mL 1.0 \
  --kcat_unit per_s \
  --conc_unit mM
'''

import argparse
import numpy as np
import pandas as pd

def mgml_to_molar(E_mg_per_mL: float, MW_kDa: float) -> float:
    """
    mg/mL -> mol/L
    1 mg/mL = 1 g/L
    MW_kDa: kDa (1 kDa = 1000 g/mol)
    """
    E_g_per_L = E_mg_per_mL
    MW_g_per_mol = MW_kDa * 1000.0
    return E_g_per_L / MW_g_per_mol

def v0_mm(kcat_per_min: float, Km: float, S0: float, E_molar: float) -> float:
    """
    v0 concentration per minute = kcat * E * S0 / (Km + S0)
    - kcat_per_min: 1/min
    - Km, S0: concentration unit consistent (e.g., mM)
    - E_molar: mol/L
    Output: concentration unit per min (same unit as S0) *but scaled by molar E*
    Note: This gives an absolute rate only if your units are physically consistent.
    For ranking, it’s still fine as long as consistent across variants.
    """
    if Km < 0 or S0 < 0 or kcat_per_min < 0 or E_molar < 0:
        return np.nan
    if S0 == 0:
        return 0.0
    return (kcat_per_min * E_molar * S0) / (Km + S0)

def rate_to_specific_activity_umol_per_min_per_mg(rate_conc_per_min: float, volume_mL: float,
                                                  enzyme_mg: float, conc_unit: str) -> float:
    """
    Convert concentration rate -> μmol/min/mg.
    If conc_unit = mM: mM = μmol/mL
      μmol/min = (mM/min) * (mL)
    If conc_unit = uM: uM = nmol/mL = 0.001 μmol/mL
      μmol/min = (uM/min) * (mL) / 1000
    """
    if enzyme_mg <= 0 or volume_mL <= 0:
        return np.nan

    cu = conc_unit.lower()
    if cu == "mm":
        umol_per_min = rate_conc_per_min * volume_mL
    elif cu == "um":
        umol_per_min = (rate_conc_per_min * volume_mL) / 1000.0
    else:
        # default treat as mM
        umol_per_min = rate_conc_per_min * volume_mL

    return umol_per_min / enzyme_mg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with sequence,kcat,Km (optional expression (mg/mL))")
    ap.add_argument("--output", required=True, help="Output CSV with v0 and specific activity")

    ap.add_argument("--PET0", type=float, default=1.0, help="Initial PET concentration (e.g., mM)")
    ap.add_argument("--E_mg_per_mL", type=float, default=1.0, help="Enzyme mg/mL (if not using expression column)")
    ap.add_argument("--use_expression_as_E", action="store_true",
                    help="Use column 'expression (mg/mL)' as enzyme mg/mL per row")

    ap.add_argument("--MW_kDa", type=float, default=30.0, help="Enzyme MW in kDa for mg/mL->molar conversion")
    ap.add_argument("--volume_mL", type=float, default=1.0, help="Assay volume (mL) for μmol/min/mg conversion")

    ap.add_argument("--kcat_unit", choices=["per_min", "per_s"], default="per_s",
                    help="Unit of kcat column. Converted to 1/min if per_s.")
    ap.add_argument("--conc_unit", choices=["mM", "uM"], default="mM",
                    help="Unit for PET0 and Km (used in conversion to μmol/min).")

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if "sequence" not in df.columns or "kcat" not in df.columns or "Km" not in df.columns:
        raise ValueError("Input must contain columns: sequence, kcat, Km")

    kcat = df["kcat"].astype(float).to_numpy()
    Km = df["Km"].astype(float).to_numpy()

    # convert to 1/min
    if args.kcat_unit == "per_s":
        kcat = kcat * 60.0

    # choose enzyme mg/mL per row
    if args.use_expression_as_E:
        expr_col = "expression (mg/mL)"
        if expr_col not in df.columns:
            raise ValueError(f"--use_expression_as_E set but '{expr_col}' not found.")
        E_mgml = df[expr_col].astype(float).to_numpy()
    else:
        E_mgml = np.full(len(df), float(args.E_mg_per_mL), dtype=float)

    # convert mg/mL -> mol/L per row
    E_molar = np.array([mgml_to_molar(e, args.MW_kDa) for e in E_mgml], dtype=float)

    # v0 (conc/min)
    v0 = np.array([v0_mm(float(kcat[i]), float(Km[i]), float(args.PET0), float(E_molar[i]))
                   for i in range(len(df))], dtype=float)

    # convert to μmol/min/mg (needs total mg enzyme = mg/mL * volume)
    enzyme_mg_total = E_mgml * float(args.volume_mL)
    spec_act = np.array([rate_to_specific_activity_umol_per_min_per_mg(float(v0[i]),
                                                                       float(args.volume_mL),
                                                                       float(enzyme_mg_total[i]),
                                                                       args.conc_unit)
                         for i in range(len(df))], dtype=float)

    out = pd.DataFrame({
        "sequence": df["sequence"].astype(str),
        "v0_rate_conc_per_min": v0,
        "activity_pred_umol_per_min_per_mg": spec_act,
    })

    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")

if __name__ == "__main__":
    main()