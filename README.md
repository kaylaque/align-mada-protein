# align-mada-protein

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

## Nimbus Approach Replication
1. Structure Data Gathering
* Tools: ColabFold/AlphaFold2, ESMFold, or AlphaFold DB, Benchling API
* Object: mutant sequences structures, target enzyme structure
* Optional: gathering wildtype structure from BLAST NCBI
2. Substrate Parameterization
* Tools: PyRosetta, structure library?
* Object: substrate structure, PARAMS file for substrate
* Optional: parameterize any cofactors or metal ions --> why tho?
3. Mechanism & Pre-Transition State Modelling
* Prerequisite Knowledge: catalytic mechanism of enzyme, key catalytic residues, reaction coordinate, transition state geomertry, the pH dependent mechanisms 
* Tools: 
    * pyrosetta : load structure pdb enzym, identify active site, calculate constraints (atom distance, dihedral, coordinate), pH specific protonation states
* Object: 
* Optional:
4. Molecular Docking
* Tools: 
    * DiffDock: get top N poses and each confidance score, visual inspections
    * PyRosetta: docking refinement protocol
5. Energy Minimization & Parameter Gathering
6. Score Normalization & Linear Combination

## Discussed Approach
1. Gathering parameter using BioPython (net charge, extinction coeff.) and pH optimum prediction (model inference)
2. Kinetic modelling approach
3. Scoring activity and Coefficient of kinetic modelling

---------------------------------------------------------------------
TASK BREAKDOWN: (from the easiest to some hassle)
- Code for parameter gathering (biopython, pyrosetta, external model)
    - run through wildtype and mutants
    - data analysis check
- Docking
- Structure data gathering
- Kinetic modelling