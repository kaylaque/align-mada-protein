import pandas as pd
import numpy as np
from pathlib import Path
import subprocess, shutil
import joblib

REPO = Path(__file__).resolve().parents[1]

TEST_CSV = REPO / "dataset" / "predictive-pet-zero-shot-test-2025.csv"
MODEL_PATH = REPO / "output" / "petml_plus_ph_activity30C_pred_v1" / "model.joblib"

PETML_ROOT = REPO / "external_models" / "PETML"
PETML_OUT  = REPO / "output" / "petml_features_test_pred30C"
OUT_SUB    = REPO / "output" / "submission_pred30C_petml_ph.csv"

PH1 = 5.5
PH2_PROXY_FOR_9 = 8.5  # clip because training max pH=8.5

def ensure_tools():
    for tool in ["hmmsearch", "mafft"]:
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing required tool in PATH: {tool}")

def run_petml(seqfile: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "petml.run", "--seqfile", str(seqfile), "--outdir", str(outdir), "--delete_temp_files", "1"]
    subprocess.run(cmd, check=True, cwd=str(PETML_ROOT))

def main():
    ensure_tools()

    df = pd.read_csv(TEST_CSV)
    if "sequence" not in df.columns:
        raise ValueError("Test CSV must have 'sequence' column.")

    df = df.copy()
    df["seq_id"] = [f"test_{i:05d}" for i in range(len(df))]

    fasta = REPO / "dataset" / "petase_test_pred30C.fasta"
    with open(fasta, "w") as f:
        for sid, seq in df[["seq_id", "sequence"]].itertuples(index=False):
            f.write(f">{sid}\n{seq}\n")

    run_petml(fasta, PETML_OUT)

    raw = pd.read_csv(PETML_OUT / "raw_scores.csv").rename(columns={"Unnamed: 0": "seq_id"})
    raw["seq_id"] = raw["seq_id"].astype(str).str.strip()
    raw = raw.rename(columns={
        "Active site HMM": "Active_site_HMM",
        "PET HMM": "PET_HMM",
        "Homologs HMM": "Homologs_HMM",
    })

    feat = df[["seq_id"]].merge(raw, on="seq_id", how="left")
    X_petml = feat[["Supervised", "Active_site_HMM", "PET_HMM", "Homologs_HMM", "Blosum"]].astype(float).fillna(0)

    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    cols = pack["feature_columns"]  # includes pH (+ maybe Tm if you trained with it)

    # If model expects Tm but we don't have it for test, set to training-median fallback.
    # We stored no median in the model pack, so we set a neutral constant.
    # If your training script USE_TM=True, you can set TM_CONST to something like 55.
    TM_CONST = 55.0

    def predict_at_ph(ph: float):
        X = X_petml.copy()
        X["pH"] = ph
        if "Tm" in cols:
            X["Tm"] = TM_CONST
        X = X.reindex(columns=cols, fill_value=0)
        pred = model.predict(X)
        return np.maximum(pred, 0)

    act1 = predict_at_ph(PH1)
    act2 = predict_at_ph(PH2_PROXY_FOR_9)

    out = df.drop(columns=["seq_id"]).copy()
    out["activity_1 (μmol [TPA]/min·mg [E])"] = act1
    out["activity_2 (μmol [TPA]/min·mg [E])"] = act2

    if "expression (mg/mL)" in out.columns:
        out["expression (mg/mL)"] = out["expression (mg/mL)"].fillna(0)

    OUT_SUB.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SUB, index=False)
    print("Wrote:", OUT_SUB)

if __name__ == "__main__":
    main()