# main_zero_shot_cluster.py (V2)
import pandas as pd
from function.scoring_cluster import compute_property_scores_clusteraware_v2

ACT1_COL = "activity_1 (μmol [TPA]/min·mg [E])"
ACT2_COL = "activity_2 (μmol [TPA]/min·mg [E])"
EXPR_COL = "expression (mg/mL)"

def main(input_csv: str, output_csv: str, delta_forge_csv: str | None):
    df = pd.read_csv(input_csv)
    assert "sequence" in df.columns

    df = compute_property_scores_clusteraware_v2(
        df,
        clusters_csv="output/param/backbone_clusters.csv",
        reps_csv="output/param/cluster_representatives.csv",
        delta_forge_csv=delta_forge_csv,
    )

    df[ACT1_COL] = df["activity_1_pred"]
    df[ACT2_COL] = df["activity_2_pred"]
    df[EXPR_COL] = df["expression_pred"]

    out = df[["sequence", ACT1_COL, ACT2_COL, EXPR_COL]]
    out.to_csv(output_csv, index=False)
    print("Wrote:", output_csv)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--delta_forge_csv", default=None, help="delta_forge_vs_wt.csv with columns sequence, delta_emb")
    args = p.parse_args()
    main(args.input, args.output, args.delta_forge_csv)

# ======================================================================
# # main_zero_shot_cluster.py
# import pandas as pd
# from scoring_cluster import compute_property_scores_clusteraware

# ACT1_COL = "activity_1 (μmol [TPA]/min·mg [E])"
# ACT2_COL = "activity_2 (μmol [TPA]/min·mg [E])"
# EXPR_COL = "expression (mg/mL)"

# def main(input_csv: str, output_csv: str):
#     df = pd.read_csv(input_csv)
#     assert "sequence" in df.columns

#     df = compute_property_scores_clusteraware(
#         df,
#         clusters_csv="backbone_clusters.csv",
#         reps_csv="cluster_representatives.csv",
#     )

#     df[ACT1_COL] = df["activity_1_pred"]
#     df[ACT2_COL] = df["activity_2_pred"]
#     df[EXPR_COL] = df["expression_pred"]

#     out = df[["sequence", ACT1_COL, ACT2_COL, EXPR_COL]]
#     out.to_csv(output_csv, index=False)

# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True)
#     p.add_argument("--output", required=True)
#     args = p.parse_args()
#     main(args.input, args.output)
