"""Signal harmonization: lead order, units, resampling, filtering, windowing, quality checks."""
from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import config as C

_LEAD_ALIASES = {
    "I": "I", "DI": "I", "LEADI": "I",
    "II": "II", "DII": "II", "LEADII": "II", "MLII": "II",
    "III": "III", "DIII": "III", "LEADIII": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
}


def normalize_lead_name(name: str) -> str | None:
    """'AVR', 'aVR', 'AV-R', 'Lead II', 'MLII', 'DII' ... -> canonical lead name (or None)."""
    key = re.sub(r"[\s_\-]", "", str(name)).upper()
    return _LEAD_ALIASES.get(key)


def to_standard_leads(signal: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """signal: (C, N) in any lead order -> (12, N) standard order + lead mask (12,) uint8.

    Missing limb leads III/aVR/aVL/aVF are derived from I and II when both are present.
    Other missing leads stay zero with mask 0.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = signal.shape[1]
    out = np.zeros((C.N_LEADS, n), dtype=np.float32)
    mask = np.zeros(C.N_LEADS, dtype=np.uint8)
    for row, nm in zip(signal, names):
        std = normalize_lead_name(nm)
        if std is not None and not mask[C.STD_LEADS.index(std)]:
            idx = C.STD_LEADS.index(std)
            out[idx] = row
            mask[idx] = 1
    i, ii = C.STD_LEADS.index("I"), C.STD_LEADS.index("II")
    if mask[i] and mask[ii]:
        derived = {"III": out[ii] - out[i], "aVR": -(out[i] + out[ii]) / 2,
                   "aVL": out[i] - out[ii] / 2, "aVF": out[ii] - out[i] / 2}
        for nm, sig in derived.items():
            idx = C.STD_LEADS.index(nm)
            if not mask[idx]:
                out[idx] = sig
                mask[idx] = 1
    return out, mask


def resample(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return signal
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(signal, frac.numerator, frac.denominator, axis=-1).astype(np.float32)


def bandpass(signal: np.ndarray, fs: float,
             low: float = C.BAND_LOW_HZ, high: float = C.BAND_HIGH_HZ) -> np.ndarray:
    nyq = fs / 2
    high = min(high, 0.9 * nyq)
    if high <= low or signal.shape[-1] < 30:
        return signal
    sos = butter(3, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def harmonize(signal: np.ndarray, fs: float, lead_names: Sequence[str],
              unit_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Native recording -> (12, N) float32 mV at STORE_FS, band-passed, plus lead mask.

    unit_scale multiplies the raw values to obtain millivolts (e.g. 0.001 for uV).
    """
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float32) * unit_scale)
    sig, mask = to_standard_leads(sig, lead_names)
    sig = resample(sig, fs, C.STORE_FS)
    sig = bandpass(sig, C.STORE_FS)
    sig[mask == 0] = 0.0
    return sig, mask


def to_windows(sig: np.ndarray, win: int = C.WINDOW_LEN, max_windows: int = 1,
               min_fraction: float = 0.8) -> list[np.ndarray]:
    """Cut a (12, N) signal into non-overlapping 10 s windows.

    A record shorter than `win` is zero-padded if it covers >= min_fraction of a window,
    otherwise it is dropped. Long recordings yield up to `max_windows` windows.
    """
    n = sig.shape[1]
    if n >= win:
        k = min(max_windows, n // win)
        return [sig[:, j * win:(j + 1) * win] for j in range(k)]
    if n >= int(min_fraction * win):
        pad = np.zeros((sig.shape[0], win), dtype=sig.dtype)
        pad[:, :n] = sig
        return [pad]
    return []


def passes_quality(window: np.ndarray, lead_mask: np.ndarray,
                   min_std_mv: float = 0.01, max_abs_mv: float = 50.0) -> bool:
    active = window[lead_mask.astype(bool)]
    if active.shape[0] == 0 or not np.isfinite(active).all():
        return False
    if np.abs(active).max() > max_abs_mv:
        return False
    flat = (active.std(axis=1) < min_std_mv).mean()
    return flat < 0.5          # reject if half or more of the active leads are flat


def signal_hash(window: np.ndarray) -> bytes:
    """Content hash used for cross-source de-duplication (a record in PTB-XL that is also in a
    Challenge bundle hashes identically if both were harmonized the same way)."""
    q = np.round(window[:, ::5] * 100).astype(np.int16)      # 100 Hz, 10 uV resolution
    return hashlib.md5(q.tobytes()).digest()


def zscore(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Per-lead z-score on (…, 12, T); all-zero (missing) leads stay zero."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)
