import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json

REPO = Path(__file__).resolve().parents[1]

CSV_PATH = REPO / "dataset" / "external_petase_activity.csv"
FEAT_PATH = REPO / "output" / "petml_features_external_activity" / "raw_scores.csv"
OUT_MODEL = REPO / "output" / "ph_preference_model.joblib"
OUT_METRICS = REPO / "output" / "ph_preference_metrics.json"

TARGET_COL = "activity_40C"  # ganti ke "activity_60C" kalau mau

df = pd.read_csv(CSV_PATH)

required = {"sample_id", "pH", TARGET_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# keep only sample_ids that have BOTH pH values
ids85 = set(df.loc[df["pH"] == 8.5, "sample_id"])
ids55 = set(df.loc[df["pH"] == 5.5, "sample_id"])
both_ids = sorted(ids85 & ids55)

sub = df[df["sample_id"].isin(both_ids)]
piv = sub.pivot_table(index="sample_id", columns="pH", values=TARGET_COL, aggfunc="first").dropna()

# label: 1 if activity at 8.5 > 5.5
y = (piv[8.5] > piv[5.5]).astype(int)

feat = pd.read_csv(FEAT_PATH)

# PETML puts sequence IDs in this column
feat = feat.rename(columns={"Unnamed: 0": "sample_id"})

# just in case there are stray spaces
feat["sample_id"] = feat["sample_id"].astype(str).str.strip()

# (optional) make feature column names safe (no spaces)
feat = feat.rename(columns={
    "Active site HMM": "Active_site_HMM",
    "PET HMM": "PET_HMM",
    "Homologs HMM": "Homologs_HMM",
})
data = piv.reset_index()[["sample_id"]].merge(feat, on="sample_id", how="left")

# numeric features only
X = data.select_dtypes(include=[np.number]).copy()
if X.isna().any().any():
    # simplest: fill missing with 0 (or median). For small data, 0 is OK baseline.
    X = X.fillna(0)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []

for tr, te in skf.split(X, y):
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    accs.append(accuracy_score(y.iloc[te], pred))
    if len(np.unique(y.iloc[te])) == 2:
        aucs.append(roc_auc_score(y.iloc[te], proba))

metrics = {
    "n_samples": int(len(y)),
    "target": TARGET_COL,
    "cv_accuracy_mean": float(np.mean(accs)),
    "cv_accuracy_std": float(np.std(accs)),
    "cv_auc_mean": float(np.mean(aucs)) if aucs else None,
    "cv_auc_std": float(np.std(aucs)) if aucs else None,
    "class_balance_y1": float(y.mean()),
}

print("Metrics:", metrics)

# fit final + save
clf.fit(X, y)
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": clf, "feature_columns": X.columns.tolist()}, OUT_MODEL)
OUT_METRICS.write_text(json.dumps(metrics, indent=2))

print("Saved model:", OUT_MODEL)
print("Saved metrics:", OUT_METRICS)