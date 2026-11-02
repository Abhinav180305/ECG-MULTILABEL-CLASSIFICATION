"""PyTorch Dataset over the unified store, with source-balanced sampling."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from . import config as C
from . import signal_utils as S
from .store import UnifiedStore


class ECGDataset(Dataset):
    """Yields dict(x=(12,T) float32, y=(L,), m=(L,), src=int).

    x is decimated from the 500 Hz store to `train_fs`, z-scored per lead. In training mode a random
    crop of `crop_sec` seconds plus light augmentation is applied; otherwise the full 10 s is returned.
    """

    def __init__(self, store: UnifiedStore, splits, train_fs: int = 100, augment: bool = False,
                 crop_sec: float = 8.0, max_samples: int | None = None, seed: int = 0):
        if C.STORE_FS % train_fs != 0:
            raise ValueError(f"train_fs must divide {C.STORE_FS}")
        self.store = store
        self.idx = store.indices(splits)
        if max_samples and len(self.idx) > max_samples:
            self.idx = np.random.default_rng(seed).choice(self.idx, max_samples, replace=False)
        self.step = C.STORE_FS // train_fs
        self.augment = augment
        self.full = C.WINDOW_LEN // self.step
        self.crop = min(int(crop_sec * train_fs), self.full)

    def __len__(self) -> int:
        return len(self.idx)

    def sample_weights(self, alpha: float = 0.5) -> np.ndarray:
        """Per-sample weights so a source with n records contributes ~ n**alpha in total (alpha=1 is
        plain uniform sampling, alpha=0 gives every source equal weight)."""
        src = self.store.source_idx[self.idx]
        counts = np.bincount(src, minlength=len(self.store.source_names)).astype(np.float64)
        w = counts[src] ** (alpha - 1.0)
        return w / w.sum()

    def make_sampler(self, alpha: float = 0.5, seed: int = 0) -> WeightedRandomSampler:
        g = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.from_numpy(self.sample_weights(alpha)), len(self), True, generator=g)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        x = np.asarray(self.store.signals[j][:, ::self.step], dtype=np.float32)      # (12, T)
        lead_mask = self.store.lead_mask[j].astype(np.float32).copy()
        if self.augment:
            rng = np.random
            s = rng.randint(0, self.full - self.crop + 1)
            x = x[:, s:s + self.crop]
            x = x * rng.uniform(0.8, 1.2) + rng.normal(0, 0.01, x.shape).astype(np.float32)
            t = np.arange(x.shape[1], dtype=np.float32) / (C.STORE_FS / self.step)
            x = x + 0.05 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 6.28))[None, :]
            drop = (rng.rand(C.N_LEADS) < 0.03) & (lead_mask > 0)       # random lead dropout
            lead_mask[drop] = 0.0
        x = S.zscore(x) * lead_mask[:, None]
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "y": torch.from_numpy(self.store.labels[j].astype(np.float32)),
            "m": torch.from_numpy(self.store.mask[j].astype(np.float32)),
            "src": int(self.store.source_idx[j]),
        }
