"""Merge many sources into one unified store: harmonize -> quality filter -> de-duplicate ->
unified labels + label mask -> patient-level split -> append to disk."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import config as C
from . import ontology as O
from . import signal_utils as S
from .errors import Cancelled
from .sources.base import BaseSource


def assign_split(source: str, patient_id: str, holdout: set[str]) -> int:
    """Deterministic patient-level split. Whole sources listed in `holdout` become external test sets."""
    if source in holdout:
        return C.SPLIT_EXTERNAL
    h = int(hashlib.md5(f"{source}:{patient_id}".encode()).hexdigest()[:8], 16) % 10
    return C.SPLIT_TRAIN if h < 8 else C.SPLIT_VAL if h == 8 else C.SPLIT_TEST


def build_store(
    sources: Iterable[BaseSource],
    out_dir: str | Path,
    holdout_sources: Iterable[str] = (),
    max_records_per_source: int | None = None,
    max_windows_per_record: int = 1,
    overwrite: bool = False,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    sources = list(sources)
    out = Path(out_dir)
    if (out / "info.json").exists() and not overwrite:
        raise FileExistsError(f"{out} already holds a finished store (use overwrite=true)")
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique, got {names}")
    holdout = set(holdout_sources)
    unknown = holdout - set(names)
    if unknown:
        raise ValueError(f"holdout_sources not among the sources: {sorted(unknown)}")

    def _total(s: BaseSource) -> int:
        n_s, cap = len(s), max_records_per_source
        return min(n_s, cap) if (cap and n_s) else (cap or n_s)

    grand = max(1, sum(_total(s) for s in sources))
    done = 0

    seen: set[bytes] = set()
    labels_l, mask_l, lead_l, age_l, sex_l, split_l, src_l = [], [], [], [], [], [], []
    meta_rows: list[tuple[str, str, str]] = []
    dropped: Counter = Counter()
    per_source: dict[str, Counter] = {n: Counter() for n in names}

    with open(out / "signals.bin", "wb") as fsig:
        for si, src in enumerate(sources):
            mask_vec = np.array(O.to_mask(src.annotated), dtype=np.uint8)
            ann = set(src.annotated)
            for k, rec in enumerate(src):
                if max_records_per_source and k >= max_records_per_source:
                    break
                if cancel is not None and cancel.is_set():
                    raise Cancelled("dataset build cancelled")
                done += 1
                if progress and done % 50 == 0:
                    progress(min(0.99, done / grand), f"{src.name}: {k + 1} records")
                try:
                    sig, lead_mask = S.harmonize(rec.signal, rec.fs, rec.lead_names, rec.unit_scale)
                except Exception:
                    dropped["harmonize_error"] += 1
                    continue
                windows = S.to_windows(sig, max_windows=max_windows_per_record)
                if not windows:
                    dropped["too_short"] += 1
                    continue
                y = np.array(O.to_multihot([l for l in rec.labels if l in ann]), dtype=np.uint8) * mask_vec
                split = assign_split(src.name, rec.patient_id, holdout)
                for wi, w in enumerate(windows):
                    if not S.passes_quality(w, lead_mask):
                        dropped["low_quality"] += 1
                        continue
                    h = S.signal_hash(w)
                    if h in seen:
                        dropped["duplicate"] += 1
                        continue
                    seen.add(h)
                    fsig.write(np.ascontiguousarray(w, dtype=np.float16).tobytes())
                    labels_l.append(y)
                    mask_l.append(mask_vec)
                    lead_l.append(lead_mask)
                    age_l.append(rec.age)
                    sex_l.append(rec.sex)
                    split_l.append(split)
                    src_l.append(si)
                    meta_rows.append((f"{rec.uid}#{wi}" if len(windows) > 1 else rec.uid, src.name, rec.patient_id))
                    per_source[src.name]["records"] += 1
                    per_source[src.name][C.SPLIT_NAMES[split]] += 1

    n = len(labels_l)
    if n == 0:
        raise RuntimeError("no records survived harmonization/quality filtering; check the source paths")
    labels = np.stack(labels_l)
    mask = np.stack(mask_l)
    np.save(out / "labels.npy", labels)
    np.save(out / "mask.npy", mask)
    np.save(out / "lead_mask.npy", np.stack(lead_l))
    np.save(out / "age.npy", np.asarray(age_l, dtype=np.float32))
    np.save(out / "sex.npy", np.asarray(sex_l, dtype=np.int8))
    np.save(out / "split.npy", np.asarray(split_l, dtype=np.int8))
    np.save(out / "source_idx.npy", np.asarray(src_l, dtype=np.int16))
    with open(out / "meta.csv", "w", encoding="utf-8") as f:
        f.write("uid,source,patient_id\n")
        for uid, s, pid in meta_rows:
            f.write(",".join('"' + str(x).replace('"', '""') + '"' for x in (uid, s, pid)) + "\n")

    prevalence = {}
    for j, lab in enumerate(O.LABELS):
        denom = int(mask[:, j].sum())
        prevalence[lab] = {"positives": int(labels[:, j].sum()), "annotated_records": denom,
                           "rate": round(float(labels[:, j].sum() / denom), 4) if denom else None}
    info = {
        "version": 1, "n": n, "labels": O.LABELS, "store_fs": C.STORE_FS, "window_len": C.WINDOW_LEN,
        "sources": names, "holdout_sources": sorted(holdout),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "dropped": dict(dropped), "label_prevalence": prevalence,
        "split_counts": {C.SPLIT_NAMES[k]: int((np.asarray(split_l) == k).sum()) for k in C.SPLIT_NAMES},
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))     # written last = "store is complete"
    if progress:
        progress(1.0, f"done: {n} records")
    return info
