# seq_features.py
import numpy as np

PKA = {
    "C_term": 3.1,
    "N_term": 8.0,
    "ASP": 3.9,
    "GLU": 4.2,
    "HIS": 6.0,
    "CYS": 8.3,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
}

AA_CHARGED = {
    "D": ("ASP", -1),
    "E": ("GLU", -1),
    "H": ("HIS", +1),
    "C": ("CYS", -1),
    "Y": ("TYR", -1),
    "K": ("LYS", +1),
    "R": ("ARG", +1),
}

HYDRO = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5,
    "C": 2.5,  "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5,  "L": 3.8,  "K": -3.9,
    "M": 1.9,  "F": 2.8,  "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

def net_charge(seq: str, pH: float) -> float:
    seq = seq.strip()
    if not seq:
        return 0.0

    nterm = PKA["N_term"]
    cterm = PKA["C_term"]
    q_n = 1 / (1 + 10**(pH - nterm))
    q_c = -1 / (1 + 10**(cterm - pH))
    total = q_n + q_c

    for aa in seq:
        if aa in AA_CHARGED:
            name, sign = AA_CHARGED[aa]
            pka = PKA[name]
            if sign > 0:
                q = 1 / (1 + 10**(pH - pka))
            else:
                q = -1 / (1 + 10**(pka - pH))
            total += q
    return total

def mean_hydrophobicity(seq: str) -> float:
    vals = [HYDRO.get(a, 0.0) for a in seq]
    return float(np.mean(vals))

def pro_gly_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    c = sum(1 for a in seq if a in ("P", "G"))
    return c / len(seq)
