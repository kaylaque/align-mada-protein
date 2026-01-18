# esm_scoring.py
import torch, os
from typing import List
import numpy as np
from esm import pretrained

# Get available CPU cores (respects container limits like Docker/Kubernetes)
try:
    # First try: use os.sched_getaffinity() (Linux-only, respects cgroups)
    num_cores = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback: use os.cpu_count() (may not respect container limits)
    num_cores = os.cpu_count() or 1

torch.set_num_threads(num_cores) # boleh disesuaikan dengan jumlah core

# Model kecil biar ringan di CPU
ESM_MODEL, ESM_ALPHABET = pretrained.esm2_t12_35M_UR50D()
ESM_MODEL.eval()
BATCH_CONVERTER = ESM_ALPHABET.get_batch_converter()

@torch.no_grad()
def esm_log_likelihood(seq: str) -> float:
    # calculate averaged ESM score of a sequence
    data = [("protein1", seq)]
    labels, strs, tokens = BATCH_CONVERTER(data)

    # CPU only – jangan .cuda()
    out = ESM_MODEL(tokens, repr_layers=[], return_contacts=False)
    logits = out["logits"]                # [B, L, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    toks = tokens[0]
    valid = (toks != ESM_ALPHABET.cls_idx) & \
            (toks != ESM_ALPHABET.eos_idx) & \
            (toks != ESM_ALPHABET.padding_idx)

    vtoks = toks[valid]
    lp = log_probs[0, valid, :]
    chosen = lp[torch.arange(lp.size(0)), vtoks]

    return float(chosen.mean().item())