"""PhysioNet/CinC-style WFDB collections whose headers carry '#Dx: <snomed,...>' comments.

Works for the Chapman-Shaoxing/Ningbo database (physionet.org/content/ecg-arrhythmia), the
CinC 2020/2021 training sets (CPSC, Georgia, PTB, PTB-XL, ...), etc. `root` is searched recursively
for *.hea files. These collections carry no patient ids, so the record name is used as patient id
(patient-level leakage cannot be excluded for them).
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import ontology as O
from .base import BaseSource, Record, parse_age, parse_sex, unit_scale_from


class PhysioNetSnomedSource(BaseSource):
    name = "physionet_snomed"

    def __init__(self, root: str, name: str = "physionet_snomed", annotated: list[str] | None = None,
                 require_labels: bool = True):
        import wfdb  # noqa: F401
        self.name = name
        self.root = Path(root)
        self.annotated = annotated or O.SNOMED_DEFAULT_ANNOTATED
        self.require_labels = require_labels
        self.headers = sorted(self.root.rglob("*.hea"))

    def __len__(self) -> int:
        return len(self.headers)

    @staticmethod
    def _comment(comments: list[str], key: str) -> str | None:
        for c in comments:
            m = re.match(rf"\s*#?\s*{key}\s*:\s*(.*)", c, flags=re.I)
            if m:
                return m.group(1).strip()
        return None

    def __iter__(self):
        import wfdb
        for hea in self.headers:
            rec_path = str(hea.with_suffix(""))
            try:
                sig, fields = wfdb.rdsamp(rec_path)
            except Exception:
                continue
            comments = fields.get("comments") or []
            dx = self._comment(comments, "Dx") or ""
            codes = [c.strip() for c in dx.split(",") if c.strip()]
            labels = sorted({O.SNOMED_MAP[c] for c in codes if c in O.SNOMED_MAP})
            if self.require_labels and not codes:
                continue
            yield Record(
                uid=hea.stem, patient_id=f"{hea.parent.name}/{hea.stem}",
                signal=sig.T, fs=float(fields["fs"]), lead_names=fields["sig_name"], labels=labels,
                age=parse_age(self._comment(comments, "Age")),
                sex=parse_sex(self._comment(comments, "Sex")),
                unit_scale=unit_scale_from((fields.get("units") or ["mV"])[0]),
            )
