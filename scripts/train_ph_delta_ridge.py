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
CSV_PATH = REPO/"dataset/external_petase_activity.csv"
FEAT_PATH = REPO/"output/petml_features_external_activity/raw_scores.csv"

TARGET_COL = "activity_40C"  # ganti ke activity_60C untuk delta60
OUTDIR = REPO/"output/ph_pref/v3_petml_reg_delta40"  # ganti v4... untuk 60C
OUT_MODEL = OUTDIR/"model.joblib"
OUT_METRICS = OUTDIR/"metrics.json"

df = pd.read_csv(CSV_PATH)

ids85 = set(df.loc[df["pH"]==8.5, "sample_id"])
ids55 = set(df.loc[df["pH"]==5.5, "sample_id"])
both = sorted(ids85 & ids55)

sub = df[df["sample_id"].isin(both)]
piv = sub.pivot_table(index="sample_id", columns="pH", values=TARGET_COL, aggfunc="first").dropna()
delta = (piv[8.5] - piv[5.5]).astype(float)  # target regression

feat = pd.read_csv(FEAT_PATH).rename(columns={"Unnamed: 0":"sample_id"})
feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
feat = feat.rename(columns={
    "Active site HMM":"Active_site_HMM",
    "PET HMM":"PET_HMM",
    "Homologs HMM":"Homologs_HMM",
})

data = piv.reset_index()[["sample_id"]].merge(feat, on="sample_id", how="left")
X = data.select_dtypes(include=[np.number]).fillna(0)
y = delta.reindex(data["sample_id"]).values

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
}

print("Metrics:", metrics)

OUTDIR.mkdir(parents=True, exist_ok=True)
model.fit(X, y)
joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, OUT_MODEL)
OUT_METRICS.write_text(json.dumps(metrics, indent=2))
print("Saved:", OUT_MODEL)