# forge_esm_scoring.py
import os
import time
import hashlib
from typing import List, Dict

import numpy as np
import torch

from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig

EMBED_CFG = LogitsConfig(sequence=True, return_embeddings=True)

def _key(seq: str, model: str) -> str:
    return hashlib.sha1((model + "|" + seq).encode("utf-8")).hexdigest()

def _to_np_fp32(x) -> np.ndarray:
    """
    Convert embeddings output (often torch tensor bf16) -> numpy float32.
    Supports shapes: (L,D) or (1,L,D).
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        # avoid torch.tensor(tensor) warning by only doing this for non-tensors
        t = torch.as_tensor(x)

    return t.detach().to(torch.float32).cpu().numpy()

def pool_mean(emb: np.ndarray) -> np.ndarray:
    """
    emb: (L,D) or (1,L,D) -> (D,)
    """
    if emb.ndim == 3:
        emb = emb[0]
    if emb.ndim != 2:
        raise ValueError(f"Expected (L,D) or (1,L,D), got shape {emb.shape}")
    return emb.mean(axis=0)

class ForgeESMEmbedder:
    def __init__(
        self,
        model: str,
        token: str,
        cache_dir: str = "cache_forge",
        max_retries: int = 5,
        backoff_sec: float = 2.0,
    ):
        self.model = model
        self.client = ESM3ForgeInferenceClient(
            model=model,
            token=token,
        )
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, seq: str) -> str:
        return os.path.join(self.cache_dir, f"{_key(seq, self.model)}.npz")

    def embed_one(self, seq: str) -> np.ndarray:
        path = self._cache_path(seq)
        if os.path.exists(path):
            return np.load(path)["emb"]

        protein = ESMProtein(sequence=str(seq))
        for attempt in range(1, self.max_retries + 1):
            try:
                pt = self.client.encode(protein)
                out = self.client.logits(pt, EMBED_CFG)
                emb = _to_np_fp32(out.embeddings)
                np.savez_compressed(path, emb=emb)
                return emb
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_sec * attempt)

        raise RuntimeError("unreachable")

    def embed_many(self, seqs: List[str]) -> Dict[str, np.ndarray]:
        """
        Cached embedding per sequence (no fancy server-side batching assumption).
        """
        out: Dict[str, np.ndarray] = {}
        for s in seqs:
            out[str(s)] = self.embed_one(str(s))
        return out

    def embed_pooled_many(self, seqs: List[str]) -> Dict[str, np.ndarray]:
        embs = self.embed_many(seqs)
        return {s: pool_mean(e) for s, e in embs.items()}