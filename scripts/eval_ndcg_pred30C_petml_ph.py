import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def ndcg_at_10(predicted_values: list, experimental_values: list, top_fraction: float = 0.1) -> float:
    predicted_values = np.array(predicted_values, dtype=float)
    experimental_values = np.array(experimental_values, dtype=float)

    assert len(predicted_values) == len(experimental_values), "Lists must be the same length"

    n_total = len(experimental_values)
    k = max(1, int(n_total * top_fraction))

    v_min, v_max = experimental_values.min(), experimental_values.max()
    if v_max == v_min:
        return 0.0
    gains = (experimental_values - v_min) / (v_max - v_min)

    predicted_order = np.argsort(predicted_values)[::-1]
    dcg = sum(
        gains[idx] / np.log2(i + 2)
        for i, idx in enumerate(predicted_order[:k])
    )

    ideal_order = np.argsort(experimental_values)[::-1]
    idcg = sum(
        gains[idx] / np.log2(i + 2)
        for i, idx in enumerate(ideal_order[:k])
    )

    return 0.0 if idcg == 0 else dcg / idcg

def load_merged_df(repo: Path) -> pd.DataFrame:
    xlsx = repo / "dataset" / "petase_activity_predicted_30C.xlsx"
    petml_raw = repo / "output" / "petml_features_pred30C" / "raw_scores.csv"

    df = pd.read_excel(xlsx)
    df = df.dropna(subset=["sample_id", "pH", "activity_30C_predicted"]).copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    feat = pd.read_csv(petml_raw).rename(columns={"Unnamed: 0": "sample_id"})
    feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
    feat = feat.rename(columns={
        "Active site HMM": "Active_site_HMM",
        "PET HMM": "PET_HMM",
        "Homologs HMM": "Homologs_HMM",
    })

    m = df.merge(feat, on="sample_id", how="inner")
    if len(m) == 0:
        raise ValueError("Merge empty. Check sample_id match between XLSX and PETML raw_scores.")
    return m

def eval_for_ph(m: pd.DataFrame, ph: float, n_splits: int = 5) -> dict:
    sub = m[m["pH"] == ph].copy()
    sub = sub.dropna(subset=["activity_30C_predicted"]).copy()

    # Features: PETML raw + pH
    X = sub[["Supervised", "Active_site_HMM", "PET_HMM", "Homologs_HMM", "Blosum"]].astype(float).fillna(0).copy()
    X["pH"] = sub["pH"].astype(float).values

    y = sub["activity_30C_predicted"].astype(float).values
    groups = sub["sample_id"].astype(str).values

    n_groups = pd.Series(groups).nunique()
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError(f"Not enough unique proteins for CV at pH={ph}. unique_proteins={n_groups}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])

    gkf = GroupKFold(n_splits=splits)
    ndcgs = []
    for tr, te in gkf.split(X, y, groups=groups):
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        nd = ndcg_at_10(pred.tolist(), y[te].tolist(), top_fraction=0.1)
        ndcgs.append(nd)

    return {
        "pH": ph,
        "n_rows": int(len(sub)),
        "n_unique_proteins": int(n_groups),
        "n_splits_used": int(splits),
        "ndcg10_mean": float(np.mean(ndcgs)),
        "ndcg10_std": float(np.std(ndcgs)),
        "folds": [float(x) for x in ndcgs],
    }

def main():
    repo = Path(__file__).resolve().parents[1]
    m = load_merged_df(repo)

    res_55 = eval_for_ph(m, ph=5.5, n_splits=5)
    res_85 = eval_for_ph(m, ph=8.5, n_splits=5)

    avg = (res_55["ndcg10_mean"] + res_85["ndcg10_mean"]) / 2.0

    print("=== NDCG@10% on predicted_30C.xlsx (target=activity_30C_predicted) ===")
    print(res_55)
    print(res_85)
    print(f"\nAverage NDCG@10% (pH 5.5 + pH 8.5)/2 = {avg:.6f}")

if __name__ == "__main__":
    main()