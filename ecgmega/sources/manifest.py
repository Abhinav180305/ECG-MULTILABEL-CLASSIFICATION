"""Generic CSV-manifest source: the adapter for ACS-ECG, ZZU pECG, EchoNext, Brugada-HUCA, your own
data, or anything else not covered by a dedicated adapter.

Manifest CSV columns (paths are relative to `root`):
    path        record path without extension (format='wfdb') or file name (format='npy')
    patient_id  optional (defaults to the path)
    age, sex    optional
    labels      optional; labels of the source separated by ';'  (e.g. "AFIB;LBBB")
    <flag cols> optional; any column listed in `flag_columns` counts as a positive label when truthy
                (useful for EchoNext-style binary columns, e.g. {"shd_moderate_or_greater_flag": "SHD"})

Options:
    label_map    {source label -> unified label}; unmapped labels are ignored
    annotated    unified labels this dataset annotates (the rest are masked out)
    format       'wfdb' (default) or 'npy' (array of shape (12, N) or (N, 12); needs `fs`)
    fs           sampling rate for 'npy'
    lead_names   lead names for 'npy' (default: standard 12-lead order)
    unit_scale   multiplier to millivolts for 'npy' (default 1.0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import ontology as O
from .base import BaseSource, Record, parse_age, parse_sex, unit_scale_from


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    try:
        return bool(v) and not (isinstance(v, float) and np.isnan(v))
    except ValueError:
        return False


class ManifestSource(BaseSource):
    name = "manifest"

    def __init__(self, root: str, manifest: str = "manifest.csv", name: str = "manifest",
                 label_map: dict[str, str] | None = None, flag_columns: dict[str, str] | None = None,
                 annotated: list[str] | None = None, format: str = "wfdb", fs: float | None = None,
                 lead_names: list[str] | None = None, unit_scale: float = 1.0, label_sep: str = ";"):
        self.name = name
        self.root = Path(root).resolve()
        mpath = (self.root / manifest).resolve()
        if not mpath.is_relative_to(self.root):
            raise ValueError("manifest must live inside root")
        self.df = pd.read_csv(mpath)
        self.label_map = label_map or {}
        self.flag_columns = flag_columns or {}
        self.annotated = annotated or O.ALL_BUT_SHD
        self.format = format
        self.fs = fs
        self.lead_names = lead_names or list(C.STD_LEADS)
        self.unit_scale = unit_scale
        self.label_sep = label_sep
        if format not in {"wfdb", "npy"}:
            raise ValueError("format must be 'wfdb' or 'npy'")
        if format == "npy" and not fs:
            raise ValueError("'fs' is required for format='npy'")

    def __len__(self) -> int:
        return len(self.df)

    def _labels(self, row) -> list[str]:
        out = []
        raw = row.get("labels")
        if isinstance(raw, str):
            for tok in raw.split(self.label_sep):
                tok = tok.strip()
                if not tok:
                    continue
                lab = self.label_map.get(tok, tok)
                if lab in O.LABEL_INDEX:
                    out.append(lab)
        for col, lab in self.flag_columns.items():
            if col in row and _truthy(row[col]) and lab in O.LABEL_INDEX:
                out.append(lab)
        return out

    def __iter__(self):
        for i, row in self.df.iterrows():
            p = (self.root / str(row["path"])).resolve()
            if not p.is_relative_to(self.root):          # never read outside the dataset root
                continue
            try:
                if self.format == "wfdb":
                    import wfdb
                    sig, fields = wfdb.rdsamp(str(p))
                    sig, fs, names = sig.T, float(fields["fs"]), fields["sig_name"]
                    scale = unit_scale_from((fields.get("units") or ["mV"])[0])
                else:
                    arr = np.load(p, allow_pickle=False)
                    sig = arr if arr.shape[0] == len(self.lead_names) else arr.T
                    fs, names, scale = float(self.fs), self.lead_names, self.unit_scale
            except Exception:
                continue
            pid = row["patient_id"] if "patient_id" in row and pd.notna(row["patient_id"]) else row["path"]
            yield Record(uid=str(row["path"]), patient_id=str(pid), signal=sig, fs=fs, lead_names=names,
                         labels=self._labels(row), age=parse_age(row.get("age")),
                         sex=parse_sex(row.get("sex")), unit_scale=scale)
