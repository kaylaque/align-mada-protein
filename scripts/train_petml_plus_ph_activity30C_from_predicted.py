import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import joblib, json

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "dataset" / "petase_activity_predicted_30C.xlsx"
PETML_RAW = REPO / "output" / "petml_features_pred30C" / "raw_scores.csv"

TARGET = "activity_30C_predicted"
USE_TM = True  # set False kalau mau PETML+pH aja

OUTDIR = REPO / "output" / "petml_plus_ph_activity30C_pred_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# load labels
df = pd.read_excel(XLSX)
df = df.dropna(subset=["sample_id", "pH", TARGET]).copy()
df["sample_id"] = df["sample_id"].astype(str).str.strip()

# load PETML features
feat = pd.read_csv(PETML_RAW).rename(columns={"Unnamed: 0": "sample_id"})
feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
feat = feat.rename(columns={
    "Active site HMM": "Active_site_HMM",
    "PET HMM": "PET_HMM",
    "Homologs HMM": "Homologs_HMM",
})

m = df.merge(feat, on="sample_id", how="inner")
if len(m) == 0:
    raise ValueError("Merge empty: sample_id between XLSX and PETML raw_scores does not match.")

X = m[["Supervised","Active_site_HMM","PET_HMM","Homologs_HMM","Blosum"]].astype(float).fillna(0).copy()
X["pH"] = m["pH"].astype(float).values
if USE_TM:
    X["Tm"] = pd.to_numeric(m["Tm"], errors="coerce").fillna(m["Tm"].median()).values

y = m[TARGET].astype(float).values
groups = m["sample_id"].values  # anti protein-leakage

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0, random_state=42)),
])

gkf = GroupKFold(n_splits=5)
spears = []
for tr, te in gkf.split(X, y, groups=groups):
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[te])
    rho = spearmanr(y[te], pred).correlation
    spears.append(0.0 if rho is None else float(rho))

metrics = {
    "TARGET": TARGET,
    "USE_TM": USE_TM,
    "n_rows": int(len(m)),
    "n_unique_proteins": int(pd.Series(groups).nunique()),
    "cv_spearman_mean": float(np.mean(spears)),
    "cv_spearman_std": float(np.std(spears)),
    "features": list(X.columns),
}

print("Metrics:", metrics)

model.fit(X, y)
joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, OUTDIR/"model.joblib")
(OUTDIR/"metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved to:", OUTDIR)