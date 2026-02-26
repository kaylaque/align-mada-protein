import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "dataset" / "cs5c03460_si_002.xlsx"

rounds = pd.read_excel(XLSX, sheet_name="Rounds").dropna(subset=["sample_id", "seq_aa"])
rounds = rounds.drop_duplicates("sample_id")[["sample_id", "seq_aa"]]
rounds["sample_id"] = rounds["sample_id"].astype(str).str.strip()

out_fa = REPO / "dataset" / "petml_rounds_sequences.fasta"
with open(out_fa, "w") as f:
    for sid, seq in rounds.itertuples(index=False):
        f.write(f">{sid}\n{seq}\n")

print("Wrote:", out_fa, "n_seq=", len(rounds))