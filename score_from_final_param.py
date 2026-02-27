#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

ACT1 = "activity_1 (μmol [TPA]/min·mg [E])"
ACT2 = "activity_2 (μmol [TPA]/min·mg [E])"
EXPR = "expression (mg/mL)"

def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)

def clip(x: pd.Series, lo=-8.0, hi=8.0) -> pd.Series:
    return x.clip(lo, hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_param_csv", required=True, help="FINAL_PARAM.csv (has sequence + features)")
    ap.add_argument("--delta_forge_csv", default=None, help="optional: delta_forge_vs_wt.csv (sequence, delta_emb)")
    ap.add_argument("--out_csv", required=True, help="submission csv")
    ap.add_argument("--aggressive", action="store_true", help="stronger weights (riskier)")
    args = ap.parse_args()

    df = pd.read_csv(args.final_param_csv)

    # drop unnamed index columns if exist
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # basic checks
    if "sequence" not in df.columns:
        raise ValueError(f"'sequence' column not found. Columns: {list(df.columns)}")

    # merge delta_forge if provided
    if args.delta_forge_csv:
        d = pd.read_csv(args.delta_forge_csv)
        if "sequence" not in d.columns:
            raise ValueError(f"delta_forge_csv must have 'sequence'. Columns: {list(d.columns)}")
        # accept delta_emb OR wt_cosine_sim
        if "delta_emb" not in d.columns and "wt_cosine_sim" not in d.columns:
            raise ValueError("delta_forge_csv must contain 'delta_emb' or 'wt_cosine_sim'")
        if "delta_emb" not in d.columns and "wt_cosine_sim" in d.columns:
            d = d.copy()
            d["delta_emb"] = 1.0 - pd.to_numeric(d["wt_cosine_sim"], errors="coerce")

        df = df.merge(d[["sequence", "delta_emb"]], on="sequence", how="left")
        # if some missing, fill neutral (median)
        if df["delta_emb"].isna().any():
            df["delta_emb"] = df["delta_emb"].fillna(df["delta_emb"].median())
    else:
        df["delta_emb"] = 0.0  # neutral if not used

    # ----- feature engineering (robust scaling) -----
    # higher score_align is generally better (closer to backbone / consistent)
    z_align = robust_z(df.get("score_align", 0.0))
    z_phopt = robust_z(df.get("phopt", 0.0))  # your pH optimality proxy
    z_instab = robust_z(df.get("instability_index", 0.0))  # lower is better
    z_mw = robust_z(df.get("molecular_weight", 0.0))        # sometimes too big hurts expression
    z_c7 = robust_z(df.get("charge7", 0.0))
    z_c9 = robust_z(df.get("charge9", 0.0))
    z_c5 = robust_z(df.get("charge5", 0.0))
    z_iep = robust_z(df.get("IEP", 0.0))

    # delta_emb: smaller better (closer to WT)
    z_delta = robust_z(df["delta_emb"])

    # ----- weights -----
    if args.aggressive:
        # sharper / higher variance
        w_act = dict(align=1.3, phopt=0.9, instab=-1.0, delta=-1.2, c7=0.25, c9=0.15, iep=0.10)
        w_expr = dict(instab=-1.4, mw=-0.8, delta=-0.6, c7=0.25, c5=0.10)
    else:
        # safer
        w_act = dict(align=1.0, phopt=0.7, instab=-0.7, delta=-0.8, c7=0.15, c9=0.10, iep=0.08)
        w_expr = dict(instab=-1.0, mw=-0.5, delta=-0.4, c7=0.15, c5=0.08)

    # ----- scoring -----
    act_score = (
        w_act["align"] * z_align +
        w_act["phopt"] * z_phopt +
        w_act["instab"] * z_instab +
        w_act["delta"] * z_delta +
        w_act["c7"] * z_c7 +
        w_act["c9"] * z_c9 +
        w_act["iep"] * z_iep
    )
    expr_score = (
        w_expr["instab"] * z_instab +
        w_expr["mw"] * z_mw +
        w_expr["delta"] * z_delta +
        w_expr["c7"] * z_c7 +
        w_expr["c5"] * z_c5
    )

    # clip to avoid insane tails (helps ranking stability)
    act1 = clip(act_score, -25, 25)
    act2 = clip(act_score * 0.98 + 0.10 * z_phopt, -25, 25)  # slight variant
    expr = clip(expr_score, -25, 25)

    out = pd.DataFrame({
        "sequence": df["sequence"].astype(str),
        ACT1: act1.astype(float),
        ACT2: act2.astype(float),
        EXPR: expr.astype(float),
    })

    # final sanity
    if out.isna().any().any():
        raise ValueError("NaN exists in output. Check your input columns / numeric coercion.")
    if not np.isfinite(out[[ACT1, ACT2, EXPR]].to_numpy()).all():
        raise ValueError("Inf exists in output.")

    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print("Rows:", len(out))

if __name__ == "__main__":
    main()