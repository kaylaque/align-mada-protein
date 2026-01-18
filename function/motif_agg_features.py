# motif_agg_features.py
import numpy as np
from typing import Tuple, Optional

# hydrophobic set (agregasi biasanya muncul saat banyak hydrophobic berdekatan)
HYDROPHOBIC = set("AILMFWVY")  # boleh tambah C kalau mau: "CAILMFWVY"

def find_gxsxg(seq: str) -> Tuple[int, int]:
    """
    Cari motif GXSXG.
    Return:
      motif_present (0/1), motif_center_index (posisi residue 'S' dalam motif), or -1 jika tidak ada.

    Kalau ada banyak motif, pilih yang "paling masuk akal":
      - S-nya berada tidak terlalu dekat ujung (avoid N/C termini)
      - paling dekat ke tengah protein (heuristik)
    """
    seq = str(seq).strip()
    L = len(seq)
    hits = []

    for i in range(L - 4):
        if seq[i] == "G" and seq[i+2] == "S" and seq[i+4] == "G":
            # motif: G X S X G
            s_pos = i + 2
            # hindari motif terlalu ujung
            if s_pos < 10 or s_pos > L - 11:
                continue
            hits.append(s_pos)

    if not hits:
        return 0, -1

    # pilih yang paling dekat ke 40% panjang (sering serine hydrolase motif agak di bagian awal-tengah)
    target = int(0.4 * L)
    best = min(hits, key=lambda p: abs(p - target))
    return 1, best


def window_charge(seq: str, center: int, window: int, charge_fn, pH: float) -> float:
    """
    Hitung net_charge pada window lokal sekitar center.
    charge_fn: fungsi net_charge(seq_sub, pH)
    """
    if center < 0:
        return 0.0
    seq = str(seq).strip()
    L = len(seq)
    half = window // 2
    lo = max(0, center - half)
    hi = min(L, center + half + 1)
    sub = seq[lo:hi]
    return float(charge_fn(sub, pH))


def max_hydrophobic_window(seq: str, win: int = 9) -> float:
    """
    Skor agregasi sederhana: maksimum fraksi hydrophobic di window sliding.
    0.0 - 1.0, makin tinggi -> makin rawan agregasi.
    """
    seq = str(seq).strip()
    L = len(seq)
    if L == 0:
        return 0.0
    if L < win:
        return sum(1 for a in seq if a in HYDROPHOBIC) / L

    best = 0.0
    for i in range(L - win + 1):
        w = seq[i:i+win]
        frac = sum(1 for a in w if a in HYDROPHOBIC) / win
        if frac > best:
            best = frac
    return float(best)


def max_hydrophobic_run(seq: str) -> int:
    """
    Panjang run hydrophobic terpanjang.
    Run panjang -> agregasi cenderung naik (untuk protein larut).
    """
    seq = str(seq).strip()
    best = 0
    cur = 0
    for a in seq:
        if a in HYDROPHOBIC:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)
