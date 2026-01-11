import pandas as pd

# df = pd.read_csv("predictive-pet-zero-shot-test-2025.csv") # tes sequence
df = pd.read_csv("pet-2025-wildtype-cds.csv") # WT sequence

# lengths = df["sequence"].astype(str).str.len() # tes sequence
lengths = df["Wt AA Sequence"].astype(str).str.len() # WT sequence

print("Total sequences:", len(lengths))
print("Min length:", lengths.min())
print("Max length:", lengths.max())
print("Unique lengths:", lengths.nunique())

# Kalau mau lihat distribusinya:
print(lengths.value_counts())
