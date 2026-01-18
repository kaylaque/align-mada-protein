# Weighted Scoring based PETase expression and activity prediction
Author: Farrel Alfaza, Kayla Queenazima, Melodia Rezadhini*, Nayaka Bagus, Sofyan Maulana
Institutions: University Gadjah Mada and Tokyo Tech Uni (*)

## Brief Description
The idea of weighting scoring is to approximately measure the combined parameters that will affect PETase protein sequence directly/indirectly on enzymatic process of TPA and its expression performance.

## Pipeline 
|- main.py
|--scoring.py (scoring and final calculation of expression and activity)
|--parameter worker
|---seq_features.py
|---esm clustering
|---blossum-alignment
|---kcat-km-prediction
|---pH-optimum-prediction

---

## 🔧 Environment Setup (CPU-only)

This project uses **Conda** to ensure a reproducible Python environment.

### Step 1 — Create the Conda environment

```bash
conda env create -f environment.yml
```

**What this does:**

* Creates a new Conda environment named **`petase_zero_shot`**
* Installs **Python 3.10** and all core scientific dependencies
  (NumPy, Pandas, SciPy, Biopython, fair-esm, etc.)

> You only need to run this **once**.

---

### Step 2 — Activate the environment

```bash
conda activate petase_zero_shot
```

**What this does:**

* Switches your shell to use the newly created environment
* Ensures that all Python packages and commands run **inside the correct environment**

> Always activate this environment **before running any scripts** in this repository.

---

### Step 3 — Install PyTorch (CPU version)

```bash
pip install -r requirements.txt
```

**What this does:**

* Installs **PyTorch (CPU-only)** and related libraries (`torchvision`, `torchaudio`)
* Uses the official **PyTorch CPU wheel index** for compatibility across systems

We install PyTorch separately because:

* PyTorch CPU wheels are distributed via a **special index URL**
* Separating it avoids common installation issues on Windows/Linux

---

## ✅ Verify Installation (Optional)

After setup, you can verify that everything works by running:

```bash
python - <<EOF
import torch
import esm
print("Torch version:", torch.__version__)
print("ESM loaded successfully")
EOF
```

If no errors appear, your environment is ready.

---

## ℹ️ Notes

* This setup is **CPU-only** and does **not require a GPU**
* For large datasets, ESM scoring may take longer on CPU
* GPU users can create a separate environment if desired

