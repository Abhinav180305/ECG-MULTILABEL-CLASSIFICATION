"""Data-pipeline tests: need only numpy/scipy/pandas (no torch, no fastapi)."""
import threading

import numpy as np
import pytest

from ecgmega import config as C
from ecgmega import ontology as O
from ecgmega import signal_utils as S
from ecgmega.builder import build_store
from ecgmega.errors import Cancelled
from ecgmega.jobs import JobManager
from ecgmega.sources import build_source
from ecgmega.sources.manifest import ManifestSource
from ecgmega.store import UnifiedStore


def test_lead_names_and_derivation():
    assert S.normalize_lead_name("AV-R") == "aVR"
    assert S.normalize_lead_name("MLII") == "II"
    assert S.normalize_lead_name("v3") == "V3"
    sig = np.random.randn(2, 500).astype(np.float32)
    out, mask = S.to_standard_leads(sig, ["I", "II"])
    assert mask.sum() == 6                                    # I, II + derived III, aVR, aVL, aVF -> 6
    np.testing.assert_allclose(out[2], sig[1] - sig[0], atol=1e-5)


def test_labels():
    assert O.labels_from_text("Abnormal ECG. Atrial fibrillation") == ["AFIB"]
    assert O.labels_from_text("Normal ECG") == ["NORM"]
    assert "CD" in O.expand_hierarchy(["LBBB"])
    assert sum(O.to_mask(["LBBB"])) == 2                      # LBBB + implied CD


def test_resample_filter_window():
    t = np.arange(0, 10, 1 / 250)
    sig = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1)).astype(np.float32)
    out, mask = S.harmonize(sig, 250, C.STD_LEADS)
    assert out.shape == (12, 5000) and mask.all()
    assert len(S.to_windows(out)) == 1
    assert S.to_windows(out[:, :2000]) == []                  # too short


def _synthetic(n, seed, **kw):
    return build_source("synthetic", f"s{seed}", options={"n": n, "seed": seed, **kw})


def test_build_store_masks_splits_dedup(tmp_path):
    a, b = _synthetic(80, 1), _synthetic(40, 2, fs=250, annotated=O.RHYTHM + O.CONDUCTION)
    dup = build_source("synthetic", "dup", options={"n": 20, "seed": 1})      # same signals as `a`
    info = build_store([a, b, dup], tmp_path / "st", holdout_sources=["s2"])
    assert info["dropped"].get("duplicate", 0) == 20
    st = UnifiedStore(tmp_path / "st")
    assert st.signals.shape == (st.n, 12, C.WINDOW_LEN)
    rows = np.flatnonzero(st.source_idx == st.source_names.index("s2"))
    mi = O.LABEL_INDEX["MI"]
    assert st.mask[rows, mi].sum() == 0 and st.labels[rows, mi].sum() == 0    # unannotated -> masked, not negative
    assert (st.split[rows] == C.SPLIT_EXTERNAL).all()
    meta = st.meta.assign(split=st.split)
    assert (meta.groupby(["source", "patient_id"])["split"].nunique() == 1).all()   # patient-level split
    with pytest.raises(FileExistsError):
        build_store([a], tmp_path / "st")


def test_build_cancel(tmp_path):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        build_store([_synthetic(10, 3)], tmp_path / "c", cancel=ev)


def test_manifest_npy_and_traversal(tmp_path):
    import pandas as pd
    x = np.random.randn(12, 2500).astype(np.float32)
    np.save(tmp_path / "a.npy", x)
    pd.DataFrame({"path": ["a.npy", "../evil.npy"], "labels": ["AF;LBBB", "AF"], "patient_id": ["p1", "p2"]}
                 ).to_csv(tmp_path / "manifest.csv", index=False)
    src = ManifestSource(str(tmp_path), name="m", label_map={"AF": "AFIB"}, format="npy", fs=250)
    recs = list(src)
    assert len(recs) == 1 and recs[0].labels == ["AFIB", "LBBB"]       # path escaping root is skipped


def test_job_manager(tmp_path):
    jm = JobManager(tmp_path)
    j = jm.submit("t", lambda ctx: (ctx.log("hi"), ctx.progress(0.5, "half"), {"ok": 1})[-1], {})
    assert j.done.wait(10)
    assert j.status == "succeeded" and j.result == {"ok": 1} and j.progress == 1.0
    bad = jm.submit("t", lambda ctx: 1 / 0, {})
    assert bad.done.wait(10)
    assert bad.status == "failed" and "ZeroDivisionError" in bad.error
    assert len(JobManager(tmp_path).list()) == 2                       # history is reloaded from disk
    ev = threading.Event()
    running = jm.submit("t", lambda ctx: (ev.wait(5), ctx.cancel.wait(5))[-1] and (_ for _ in ()).throw(Cancelled()), {})
    assert jm.cancel(running.id) and running.done.wait(10)
    assert running.status == "cancelled"
    ev.set()
    jm.shutdown()
