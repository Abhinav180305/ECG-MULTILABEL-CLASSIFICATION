"""MIMIC-IV-ECG (physionet.org/content/mimic-iv-ecg). `root` holds record_list.csv,
machine_measurements.csv and the waveform folders. Labels come from the machine reports via
keyword rules (ontology.labels_from_text) and are therefore noisier than cardiologist labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import ontology as O
from .base import BaseSource, Record, unit_scale_from


class MimicECGSource(BaseSource):
    name = "mimic_ecg"

    def __init__(self, root: str, name: str = "mimic_ecg", annotated: list[str] | None = None):
        import wfdb  # noqa: F401
        self.name = name
        self.root = Path(root)
        self.annotated = annotated or O.TEXT_DEFAULT_ANNOTATED
        rl = pd.read_csv(self.root / "record_list.csv")
        mm = pd.read_csv(self.root / "machine_measurements.csv")
        report_cols = [c for c in mm.columns if c.startswith("report_")]
        mm["_text"] = mm[report_cols].fillna("").astype(str).agg(" ".join, axis=1)
        self.df = rl.merge(mm[["study_id", "_text"]], on="study_id", how="inner")

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        import wfdb
        for _, row in self.df.iterrows():
            try:
                sig, fields = wfdb.rdsamp(str(self.root / row["path"]))
            except Exception:
                continue
            yield Record(
                uid=str(row["study_id"]), patient_id=str(row["subject_id"]),
                signal=sig.T, fs=float(fields["fs"]), lead_names=fields["sig_name"],
                labels=O.labels_from_text(row["_text"]),
                unit_scale=unit_scale_from((fields.get("units") or ["mV"])[0]),
            )
