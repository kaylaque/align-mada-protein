import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "dataset" / "petase_activity_predicted_30C.xlsx"
OUT_FA = REPO / "dataset" / "petase_pred30C_sequences.fasta"

df = pd.read_excel(XLSX)
df = df.dropna(subset=["sample_id", "seq_aa"]).drop_duplicates("sample_id")[["sample_id", "seq_aa"]]
df["sample_id"] = df["sample_id"].astype(str).str.strip()

OUT_FA.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FA, "w") as f:
    for sid, seq in df.itertuples(index=False):
        f.write(f">{sid}\n{seq}\n")

print("Wrote:", OUT_FA, "n_seq=", len(df))