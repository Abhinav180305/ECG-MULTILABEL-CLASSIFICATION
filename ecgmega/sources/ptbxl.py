"""PTB-XL (https://physionet.org/content/ptb-xl/). Point `root` at the folder holding
ptbxl_database.csv and scp_statements.csv."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from .. import ontology as O
from .base import BaseSource, Record, parse_age, parse_sex, unit_scale_from


class PTBXLSource(BaseSource):
    name = "ptbxl"
    annotated = O.ALL_BUT_SHD

    def __init__(self, root: str, use_hr: bool = True, min_likelihood: float = 1.0, name: str = "ptbxl"):
        import wfdb  # noqa: F401  (fail early if missing)
        self.name = name
        self.root = Path(root)
        self.use_hr = use_hr
        self.min_likelihood = min_likelihood
        self.df = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        self.df["scp_codes"] = self.df["scp_codes"].apply(ast.literal_eval)
        stm = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        diag = stm[stm["diagnostic"] == 1]
        self.diag_class = diag["diagnostic_class"].to_dict()

    def __len__(self) -> int:
        return len(self.df)

    def _labels(self, scp: dict) -> list[str]:
        out: set[str] = set()
        for code, lik in scp.items():
            if lik < self.min_likelihood:
                continue
            if code in O.PTBXL_CODE_MAP:
                out.add(O.PTBXL_CODE_MAP[code])
            cls = self.diag_class.get(code)
            if cls in O.PTBXL_SUPERCLASS:
                out.add(cls)
        return sorted(out)

    def __iter__(self):
        import wfdb
        col = "filename_hr" if self.use_hr else "filename_lr"
        for ecg_id, row in self.df.iterrows():
            try:
                sig, fields = wfdb.rdsamp(str(self.root / row[col]))
            except Exception:                      # corrupt / missing file: skip
                continue
            yield Record(
                uid=str(ecg_id), patient_id=str(int(row["patient_id"])),
                signal=sig.T, fs=float(fields["fs"]), lead_names=fields["sig_name"],
                labels=self._labels(row["scp_codes"]),
                age=parse_age(row.get("age")), sex=parse_sex(row.get("sex")),
                unit_scale=unit_scale_from((fields.get("units") or ["mV"])[0]),
            )
