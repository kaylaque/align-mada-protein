# esm_cache.py
import os
import numpy as np
import pandas as pd
from typing import Dict
from function.esm_scoring import esm_log_likelihood

CACHE_PATH = "esm_ll_cache.npz"

def load_cache(path: str = CACHE_PATH) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    keys = data["keys"].tolist()
    vals = data["vals"].tolist()
    return dict(zip(keys, vals))

def save_cache(cache: Dict[str, float], path: str = CACHE_PATH) -> None:
    keys = np.array(list(cache.keys()), dtype=object)
    vals = np.array(list(cache.values()), dtype=np.float32)
    np.savez_compressed(path, keys=keys, vals=vals)

def esm_ll_for_sequences(seqs, cache_path: str = CACHE_PATH) -> np.ndarray:
    cache = load_cache(cache_path)
    out = np.zeros(len(seqs), dtype=np.float32)

    updated = False
    for i, s in enumerate(seqs):
        s = str(s)
        if s in cache:
            out[i] = cache[s]
        else:
            val = float(esm_log_likelihood(s))
            cache[s] = val
            out[i] = val
            updated = True

    if updated:
        save_cache(cache, cache_path)

    return out