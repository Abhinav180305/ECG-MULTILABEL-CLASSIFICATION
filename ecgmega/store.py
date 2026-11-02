"""Reader for the unified on-disk store written by builder.build_store.

Layout of a store directory (plain numpy, memory-mapped, safe with many DataLoader workers):
    signals.bin   float16, shape (N, 12, 5000)   mV @ 500 Hz, band-passed, standard lead order
    labels.npy    uint8   (N, L)    multi-hot over ontology.LABELS (already * mask)
    mask.npy      uint8   (N, L)    1 where the record's source annotates that label
    lead_mask.npy uint8   (N, 12)   1 where a lead is real/derived (0 = zero-filled)
    age.npy float32 (N,), sex.npy int8 (N,), split.npy int8 (N,)   0 train, 1 val, 2 test, 3 external
    meta.csv      uid, source, patient_id
    info.json     summary statistics + the label list the arrays were built with
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class UnifiedStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        info_file = self.path / "info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"{self.path} is not a finished store (no info.json)")
        self.info = json.loads(info_file.read_text())
        self.n = int(self.info["n"])
        self.labels = np.load(self.path / "labels.npy")
        self.mask = np.load(self.path / "mask.npy")
        self.lead_mask = np.load(self.path / "lead_mask.npy")
        self.age = np.load(self.path / "age.npy")
        self.sex = np.load(self.path / "sex.npy")
        self.split = np.load(self.path / "split.npy")
        self.source_names: list[str] = list(self.info["sources"])
        self.source_idx = np.load(self.path / "source_idx.npy")
        self._signals = None
        self._meta = None

    # memmaps are re-opened lazily in each process (cheap and spawn-safe)
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_signals"] = None
        d["_meta"] = None
        return d

    @property
    def signals(self) -> np.ndarray:
        if self._signals is None:
            self._signals = np.memmap(self.path / "signals.bin", dtype=np.float16, mode="r",
                                      shape=(self.n, C.N_LEADS, C.WINDOW_LEN))
        return self._signals

    @property
    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self._meta = pd.read_csv(self.path / "meta.csv", dtype=str)
        return self._meta

    def indices(self, splits: list[int] | tuple[int, ...]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.split, list(splits)))

    def __len__(self) -> int:
        return self.n
