"""Synthetic 12-lead ECG generator for smoke tests / demos (no real data needed).

The waveforms are crude but class-dependent (heart rate, RR regularity, QRS width, ST shift,
amplitude), so a model can actually learn them. NOT physiologically accurate.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from .. import ontology as O
from .base import BaseSource, Record

SYNTH_CLASSES = ["NORM", "AFIB", "SBRAD", "STACH", "LBBB", "RBBB", "MI", "STTC", "HYP"]
_LEAD_GAIN = np.array([1.0, 1.3, 0.5, -1.0, 0.4, 0.9, 0.3, 0.6, 1.0, 1.3, 1.1, 0.8])


def _g(t, mu, sd, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def synth_record(label: str, fs: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(fs * dur)
    t = np.arange(n) / fs
    hr = {"SBRAD": 45, "STACH": 125}.get(label, 72) + rng.normal(0, 4)
    rr = 60.0 / hr
    beats, cur = [], rng.uniform(0.2, 0.6)
    while cur < dur:
        beats.append(cur)
        jitter = rng.normal(0, 0.18 * rr) if label == "AFIB" else rng.normal(0, 0.01)
        cur += max(0.3, rr + jitter)
    qrs_sd = 0.04 if label in {"LBBB", "RBBB"} else 0.011
    r_amp = 1.9 if label == "HYP" else 1.0
    st = 0.25 if label in {"MI", "STTC"} else 0.0
    base = np.zeros(n)
    for b in beats:
        if label != "AFIB":
            base += _g(t, b - 0.16, 0.025, 0.12)                    # P wave
        base += _g(t, b, qrs_sd, r_amp)                              # R
        base += _g(t, b - 0.03, 0.012, -0.15) + _g(t, b + 0.03, 0.012, -0.2)   # Q, S
        if label == "RBBB":
            base += _g(t, b + 0.09, 0.03, 0.5)                       # R'
        if label == "MI":
            base += _g(t, b - 0.035, 0.02, -0.45)                    # pathological Q
        base += _g(t, b + 0.09 + 0.05, 0.05, st)                     # ST shift
        base += _g(t, b + 0.28, 0.06, 0.3 if label != "STTC" else -0.3)   # T wave
    if label == "AFIB":
        base += 0.05 * np.sin(2 * np.pi * rng.uniform(5, 7) * t + rng.uniform(0, 6.28))   # f-waves
    gains = _LEAD_GAIN * rng.uniform(0.85, 1.15, size=C.N_LEADS)
    sig = gains[:, None] * base[None, :]
    wander = 0.1 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.28))
    sig += wander[None, :] + rng.normal(0, 0.02, size=sig.shape)
    return sig.astype(np.float32)


class SyntheticSource(BaseSource):
    name = "synthetic"

    def __init__(self, n: int = 200, seed: int = 0, fs: int = 500, duration: float = 10.0,
                 name: str = "synthetic", annotated: list[str] | None = None, root: str | None = None):
        self.name = name
        self.n, self.seed, self.fs, self.duration = int(n), int(seed), int(fs), float(duration)
        self.annotated = annotated or list(O.ALL_BUT_SHD)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pid, left = 0, 0
        for i in range(self.n):
            if left == 0:
                pid += 1
                left = int(rng.integers(1, 4))
            left -= 1
            lab = SYNTH_CLASSES[int(rng.integers(len(SYNTH_CLASSES)))]
            yield Record(uid=f"{self.name}-{i}", patient_id=f"{self.name}-p{pid}",
                         signal=synth_record(lab, self.fs, self.duration, rng), fs=float(self.fs),
                         lead_names=list(C.STD_LEADS), labels=[lab],
                         age=float(rng.integers(20, 90)), sex=int(rng.integers(0, 2)))
