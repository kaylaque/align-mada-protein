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

## Author Contributions

This project was developed collaboratively by a team from Universitas 
Gadjah Mada, in collaboration with Tokyo Institute of Technology 
(Melodia Rezadhini).

**Kayla Queenazima** (UGM) — 
Conceptualised the weighted scoring framework integrating multiple 
biological parameters; designed the overall pipeline architecture 
combining ESM embeddings, BLOSUM alignment scoring, and kcat/Km 
prediction; managed submission workflow and final parameter tuning.

**Sofyan Maulana** (UGM) — 
Implemented ESM zero-shot scoring module (forge integration) for 
PETase sequence embedding generation; developed delta-score analysis 
comparing engineered variants against wild-type.

**Farrel Alfaza, Nayaka Bagus** (UGM), **Melodia Rezadhini** (Tokyo Institute of Technology) — 
Data curation and pre-processing of PETase sequence datasets; biological validation of computational predictions; cross-checking of scoring outputs against known enzymatic behaviour and structural constraints; experimental context interpretation; External validation; biological plausibility assessment of predicted sequences from a protein engineering perspective; advisory input on PETase enzymatic mechanism and structural-functional relationships.

## Context

This work was developed as a submission to the **alignBio PETase Protein 
Engineering Tournament** (https://alignbio.org/benchmarks/) — a community 
benchmark for ML-based prediction of PETase enzyme variants. The pipeline 
combines ESM protein language models with weighted feature scoring to 
predict expression and activity of engineered PETase sequences. 

Note: As of writing, the benchmark organisers have not announced the 
tournament results.
