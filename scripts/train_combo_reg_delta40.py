import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib, json

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "dataset" / "external_petase_activity.csv"
FEAT_PATH = REPO / "output" / "petml_features_external_activity" / "raw_scores.csv"

TARGET_COL = "activity_40C"
OUTDIR = REPO / "output" / "ph_pref" / "v8_combo_reg_delta40"
OUT_MODEL = OUTDIR / "model.joblib"

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def seq_features(seq: str) -> dict:
    s = str(seq).strip().upper()
    L = len(s)
    if L == 0:
        return {"len": 0, **{f"frac_{a}": 0.0 for a in AA20},
                "frac_acidic": 0.0, "frac_basic": 0.0, "net_charge_proxy": 0.0}
    counts = {a: s.count(a) for a in AA20}
    feats = {"len": float(L)}
    feats.update({f"frac_{a}": counts[a] / L for a in AA20})
    acidic = (counts["D"] + counts["E"]) / L
    basic = (counts["K"] + counts["R"] + counts["H"]) / L
    feats["frac_acidic"] = acidic
    feats["frac_basic"] = basic
    feats["net_charge_proxy"] = (counts["K"] + counts["R"] - counts["D"] - counts["E"]) / L
    return feats

df = pd.read_csv(CSV_PATH)

ids85 = set(df.loc[df["pH"] == 8.5, "sample_id"])
ids55 = set(df.loc[df["pH"] == 5.5, "sample_id"])
both = sorted(ids85 & ids55)

sub = df[df["sample_id"].isin(both)]
piv = sub.pivot_table(index="sample_id", columns="pH", values=TARGET_COL, aggfunc="first").dropna()
delta = (piv[8.5] - piv[5.5]).astype(float)

seq_df = df.dropna(subset=["sample_id", "seq_aa"]).drop_duplicates("sample_id")[["sample_id", "seq_aa"]]
seq_df["sample_id"] = seq_df["sample_id"].astype(str).str.strip()

use = pd.DataFrame({"sample_id": piv.index.astype(str)}).merge(seq_df, on="sample_id", how="left")
if use["seq_aa"].isna().any():
    missing = use[use["seq_aa"].isna()]["sample_id"].tolist()[:10]
    raise ValueError(f"Missing seq_aa for some paired sample_id. Example missing: {missing}")

X_seq = pd.DataFrame([seq_features(s) for s in use["seq_aa"]])
y = delta.reindex(use["sample_id"]).values

feat = pd.read_csv(FEAT_PATH).rename(columns={"Unnamed: 0":"sample_id"})
feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
feat = feat.rename(columns={
    "Active site HMM":"Active_site_HMM",
    "PET HMM":"PET_HMM",
    "Homologs HMM":"Homologs_HMM",
})

use2 = pd.DataFrame({"sample_id": use["sample_id"]}).merge(feat, on="sample_id", how="left")
X_petml = use2.select_dtypes(include=[np.number]).fillna(0)

X = pd.concat([X_petml.reset_index(drop=True), X_seq.reset_index(drop=True)], axis=1)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0, random_state=42)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, rmses = [], []
for tr, te in kf.split(X):
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[te])
    maes.append(mean_absolute_error(y[te], pred))
    rmses.append(np.sqrt(mean_squared_error(y[te], pred)))

metrics = {
    "n_samples": int(len(y)),
    "target": TARGET_COL,
    "cv_mae_mean": float(np.mean(maes)),
    "cv_mae_std": float(np.std(maes)),
    "cv_rmse_mean": float(np.mean(rmses)),
    "cv_rmse_std": float(np.std(rmses)),
    "n_features": int(X.shape[1]),
}

print("Metrics:", metrics)

OUTDIR.mkdir(parents=True, exist_ok=True)
model.fit(X, y)
joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, OUT_MODEL)
(OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved:", OUT_MODEL)