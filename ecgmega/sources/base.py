"""Source interface: every dataset adapter yields `Record`s in its native format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class Record:
    uid: str                         # unique within the source
    patient_id: str                  # used for patient-level splitting
    signal: np.ndarray               # (n_leads, n_samples) native sampling rate
    fs: float
    lead_names: Sequence[str]
    labels: list[str]                # unified labels (see ontology.LABELS)
    age: float = float("nan")
    sex: int = -1                    # 0 = male, 1 = female, -1 = unknown
    unit_scale: float = 1.0          # multiply raw values by this to get millivolts
    meta: dict = field(default_factory=dict)


class BaseSource:
    name: str = "base"
    annotated: list[str] = []        # labels this source reliably annotates (others are masked)

    def __len__(self) -> int:        # best-effort count, used for progress reporting
        return 0

    def __iter__(self) -> Iterator[Record]:
        raise NotImplementedError


def parse_sex(v) -> int:
    s = str(v).strip().lower()
    if s in {"0", "m", "male", "man"}:
        return 0
    if s in {"1", "f", "female", "woman"}:
        return 1
    return -1


def parse_age(v) -> float:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return a if 0 <= a <= 120 else float("nan")


def unit_scale_from(units: str | None) -> float:
    """WFDB unit string -> factor that converts to millivolts."""
    u = (units or "mV").strip().lower()
    return {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "v": 1e3, "nv": 1e-6}.get(u, 1.0)
