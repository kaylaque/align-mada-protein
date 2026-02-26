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
XLSX = REPO / "dataset" / "cs5c03460_si_002.xlsx"
PETML_RAW = REPO / "output" / "petml_features_rounds" / "raw_scores.csv"

TEMP = 40
TARGET = "umol_product_per_mg_enzyme"   # kamu bisa ganti ke kolom lain kalau perlu
OUTDIR = REPO / "output" / "petml_plus_ph_activity40C_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# load activity @30C
act = pd.read_excel(XLSX, sheet_name="Activity")
act = act.dropna(subset=["sample_id", "temperature_c", "pH", TARGET]).copy()
act = act[act["sample_id"].astype(str).str.lower() != "blank"]
act = act[act["temperature_c"] == TEMP].copy()
act["sample_id"] = act["sample_id"].astype(str).str.strip()

# load PETML raw scores
feat = pd.read_csv(PETML_RAW).rename(columns={"Unnamed: 0": "sample_id"})
feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
feat = feat.rename(columns={
    "Active site HMM": "Active_site_HMM",
    "PET HMM": "PET_HMM",
    "Homologs HMM": "Homologs_HMM",
})

df = act.merge(feat, on="sample_id", how="inner")

# features = PETML + pH (30C fixed, so no temp feature needed)
X = df[["Supervised", "Active_site_HMM", "PET_HMM", "Homologs_HMM", "Blosum"]].astype(float).fillna(0).copy()
X["pH"] = df["pH"].astype(float).values

y = df[TARGET].astype(float).values
groups = df["sample_id"].astype(str).values  # avoid protein leakage

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
    "TEMP": TEMP,
    "TARGET": TARGET,
    "n_rows": int(len(df)),
    "n_unique_proteins": int(pd.Series(groups).nunique()),
    "cv_spearman_mean": float(np.mean(spears)),
    "cv_spearman_std": float(np.std(spears)),
    "features": list(X.columns),
}

print("Metrics:", metrics)

model.fit(X, y)
joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, OUTDIR / "model.joblib")
(OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved to:", OUTDIR)