# main_zero_shot.py
# Cara pakai:
'''
python main_zero_shot.py \
  --input predictive-pet-zero-shot-test-2025.csv \
  --output petase_zero_shot_submission_baseline.csv
  
'''

import pandas as pd
from scoring import compute_property_scores

ACT1_COL = "activity_1 (μmol [TPA]/min·mg [E])"
ACT2_COL = "activity_2 (μmol [TPA]/min·mg [E])"
EXPR_COL = "expression (mg/mL)"

def main(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    assert "sequence" in df.columns, "Kolom 'sequence' tidak ditemukan"

    # Hitung skor zero-shot
    df = compute_property_scores(df)

    # Map skor kita ke nama kolom resmi kompetisi
    df[ACT1_COL] = df["activity_1_pred"]
    df[ACT2_COL] = df["activity_2_pred"]
    df[EXPR_COL] = df["expression_pred"]

    # Urutan kolom: sama seperti file asli (sequence + 3 property)
    out = df[["sequence", ACT1_COL, ACT2_COL, EXPR_COL]]

    out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="predictive-pet-zero-shot-test-2025.csv")
    p.add_argument("--output", required=True,
                   help="nama file output submission")
    args = p.parse_args()
    main(args.input, args.output)
