import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
csv_path = REPO / "dataset" / "external_petase_activity.csv"
out_fa = REPO / "dataset" / "external_activity_sequences.fasta"

df = pd.read_csv(csv_path)

need_cols = {"sample_id", "seq_aa"}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

seq_df = df.dropna(subset=["sample_id", "seq_aa"]).drop_duplicates("sample_id")[["sample_id", "seq_aa"]]

out_fa.parent.mkdir(parents=True, exist_ok=True)
with open(out_fa, "w") as f:
    for sid, seq in seq_df.itertuples(index=False):
        f.write(f">{sid}\n{seq}\n")

print(f"Wrote: {out_fa} | n_seq={len(seq_df)}")