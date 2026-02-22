import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib, json

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "dataset" / "external_petase_activity.csv"

TARGET_COL = "activity_40C"
OUTDIR = REPO / "output" / "ph_pref" / "v5_seqfeat_cls_40C"
OUT_MODEL = OUTDIR / "model.joblib"
OUT_METRICS = OUTDIR / "metrics.json"

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

# build paired dataset
ids85 = set(df.loc[df["pH"] == 8.5, "sample_id"])
ids55 = set(df.loc[df["pH"] == 5.5, "sample_id"])
both = sorted(ids85 & ids55)

sub = df[df["sample_id"].isin(both)]
piv = sub.pivot_table(index="sample_id", columns="pH", values=TARGET_COL, aggfunc="first").dropna()
y = (piv[8.5] > piv[5.5]).astype(int)

# get one sequence per sample_id
seq_df = df.dropna(subset=["sample_id", "seq_aa"]).drop_duplicates("sample_id")[["sample_id", "seq_aa"]]
seq_df["sample_id"] = seq_df["sample_id"].astype(str).str.strip()

use = pd.DataFrame({"sample_id": piv.index.astype(str)}).merge(seq_df, on="sample_id", how="left")
if use["seq_aa"].isna().any():
    missing = use[use["seq_aa"].isna()]["sample_id"].tolist()[:10]
    raise ValueError(f"Missing seq_aa for some paired sample_id. Example missing: {missing}")

X = pd.DataFrame([seq_features(s) for s in use["seq_aa"]])
y = y.reindex(use["sample_id"]).values

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []
for tr, te in skf.split(X, y):
    clf.fit(X.iloc[tr], y[tr])
    pred = clf.predict(X.iloc[te])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    accs.append(accuracy_score(y[te], pred))
    if len(np.unique(y[te])) == 2:
        aucs.append(roc_auc_score(y[te], proba))

metrics = {
    "n_samples": int(len(y)),
    "target": TARGET_COL,
    "cv_accuracy_mean": float(np.mean(accs)),
    "cv_accuracy_std": float(np.std(accs)),
    "cv_auc_mean": float(np.mean(aucs)) if aucs else None,
    "cv_auc_std": float(np.std(aucs)) if aucs else None,
    "class_balance_y1": float(np.mean(y)),
    "features": list(X.columns),
}

print("Metrics:", metrics)

OUTDIR.mkdir(parents=True, exist_ok=True)
clf.fit(X, y)
joblib.dump({"model": clf, "feature_columns": X.columns.tolist()}, OUT_MODEL)
(OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved:", OUT_MODEL)