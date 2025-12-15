import pandas as pd

df = pd.read_csv("predictive-pet-zero-shot-test-2025.csv")

lengths = df["sequence"].astype(str).str.len()

print("Total sequences:", len(lengths))
print("Min length:", lengths.min())
print("Max length:", lengths.max())
print("Unique lengths:", lengths.nunique())

# Kalau mau lihat distribusinya:
print(lengths.value_counts())
