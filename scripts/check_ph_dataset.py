import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO/"dataset/external_petase_activity.csv")

def summarize(target_col):
    ids85 = set(df.loc[df["pH"]==8.5, "sample_id"])
    ids55 = set(df.loc[df["pH"]==5.5, "sample_id"])
    both = sorted(ids85 & ids55)
    sub = df[df["sample_id"].isin(both)]
    piv = sub.pivot_table(index="sample_id", columns="pH", values=target_col, aggfunc="first").dropna()

    y = (piv[8.5] > piv[5.5]).astype(int)
    delta = piv[8.5] - piv[5.5]

    print("\n===", target_col, "===")
    print("n_pairs:", len(piv))
    print("class_balance(y=1):", float(y.mean()))
    print("delta mean/std:", float(delta.mean()), float(delta.std()))
    print("delta quartiles:", np.quantile(delta, [0.1,0.25,0.5,0.75,0.9]))

summarize("activity_40C")
if "activity_60C" in df.columns:
    summarize("activity_60C")