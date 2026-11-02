"""FastAPI service: build the unified dataset, train, inspect runs, and serve predictions.

Run:  uvicorn ecgmega.api:app --host 0.0.0.0 --port 8000
Env:  ECG_WORKSPACE, ECG_ALLOWED_ROOTS, ECG_API_KEY, ECG_CHECKPOINT, ECG_MAX_UPLOAD_MB  (see config.py)
"""
from __future__ import annotations

import hmac
import io
import json
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from . import config as C
from . import ontology as O
from .jobs import JobContext, JobManager
from .sources import KINDS, build_source
from .train import TrainConfig, TrainHyper

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
LABEL_DESCRIPTIONS = {
    "NORM": "Normal ECG / sinus rhythm", "MI": "Myocardial infarction", "STTC": "ST/T change",
    "CD": "Conduction disturbance", "HYP": "Hypertrophy", "AFIB": "Atrial fibrillation",
    "AFLT": "Atrial flutter", "SBRAD": "Sinus bradycardia", "STACH": "Sinus tachycardia",
    "SARRH": "Sinus arrhythmia", "LBBB": "Left bundle branch block", "RBBB": "Right bundle branch block",
    "IAVB": "First-degree AV block", "PVC": "Premature ventricular contraction",
    "PAC": "Premature atrial/supraventricular contraction", "LAFB": "Left anterior fascicular block",
    "PACE": "Paced rhythm", "SHD": "Structural heart disease (echo-derived)",
}


# ------------------------------------------------------------------ helpers
def _name(v: str, what: str) -> str:
    if not NAME_RE.match(v):
        raise HTTPException(422, f"invalid {what} '{v}': use letters, digits, '_', '-', '.' (max 64)")
    return v


def _allowed_path(p: str) -> str:
    rp = Path(p).expanduser().resolve()
    roots = [C.WORKSPACE, *C.ALLOWED_ROOTS]
    if not any(rp == r or rp.is_relative_to(r) for r in roots):
        raise HTTPException(403, "path is outside the allowed roots; set ECG_ALLOWED_ROOTS on the server")
    return str(rp)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if C.API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, C.API_KEY)):
        raise HTTPException(401, "missing or invalid X-API-Key")


def _store_dir(name: str) -> Path:
    return C.WORKSPACE / "stores" / _name(name, "store name")


def _run_dir(name: str) -> Path:
    return C.WORKSPACE / "runs" / _name(name, "run name")


# ------------------------------------------------------------------ request models
class SourceSpec(BaseModel):
    name: str = Field(..., description="unique source name, e.g. 'ptbxl'")
    kind: Literal["ptbxl", "physionet_snomed", "mimic_ecg", "manifest", "synthetic"]
    root: str | None = Field(None, description="server-side dataset folder (must be inside the allowed roots)")
    options: dict = Field(default_factory=dict, description="kind-specific options, see sources/*.py")


class BuildRequest(BaseModel):
    store_name: str
    sources: list[SourceSpec] = Field(..., min_length=1)
    holdout_sources: list[str] = Field(default_factory=list, description="sources kept entirely for external testing")
    max_records_per_source: int | None = Field(None, ge=1)
    max_windows_per_record: int = Field(1, ge=1, le=50)
    overwrite: bool = False


class TrainRequest(BaseModel):
    store_name: str
    run_name: str
    hyper: TrainHyper = Field(default_factory=TrainHyper)
    resume_from_run: str | None = Field(None, description="initialise from another run's best.pt")
    activate: bool = Field(True, description="load the finished model into the prediction endpoints")


class LoadModelRequest(BaseModel):
    run_name: str


class PredictJSON(BaseModel):
    signal: list[list[float]] = Field(..., description="2-D array of samples in mV (or uV, see unit)")
    fs: float = Field(..., gt=10, le=10000)
    lead_names: list[str] | None = None
    orientation: Literal["leads_first", "samples_first"] = "leads_first"
    unit: Literal["mV", "uV"] = "mV"


# ------------------------------------------------------------------ app state / lifespan
class State:
    jobs: JobManager
    predictor = None


state = State()


def load_predictor(path: str) -> dict:
    from .inference import Predictor        # imported lazily: needs torch
    state.predictor = Predictor(path)
    return state.predictor.info()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for sub in ("stores", "runs", "jobs"):
        (C.WORKSPACE / sub).mkdir(parents=True, exist_ok=True)
    state.jobs = JobManager(C.WORKSPACE / "jobs")
    ckpt = C.DEFAULT_CHECKPOINT
    if not ckpt:                                       # fall back to the most recent run's best.pt
        cands = sorted((C.WORKSPACE / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        ckpt = str(cands[-1]) if cands else ""
    if ckpt and Path(ckpt).exists():
        try:
            load_predictor(ckpt)
        except Exception as e:                         # noqa: BLE001
            print(f"[ecgmega] could not load checkpoint {ckpt}: {e}")
    yield
    state.jobs.shutdown()


app = FastAPI(title="ECG Mega", version=__version__, lifespan=lifespan,
              description="Unify public ECG datasets, train one multi-label model, and serve predictions. "
                          + C.DISCLAIMER)
public = APIRouter()
secured = APIRouter(dependencies=[Depends(require_key)])


# ------------------------------------------------------------------ public routes
INDEX_HTML = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ECG Mega</title>
<style>body{font:15px system-ui;max-width:760px;margin:2rem auto;padding:0 16px;line-height:1.5}
table{border-collapse:collapse;width:100%}td,th{padding:4px 8px;border-bottom:1px solid #8884;text-align:left}
.bar{height:8px;background:#2b7;border-radius:4px}code{background:#8882;padding:1px 4px;border-radius:3px}
.warn{background:#fb03;padding:8px 12px;border-radius:6px}</style>
<h1>ECG Mega</h1><p class=warn>Research use only. Not a medical device.</p>
<p>Upload a 12-lead recording: <code>.npy</code> / <code>.csv</code>, or a WFDB pair (<code>.hea</code> + <code>.dat</code>).
API docs: <a href=/docs>/docs</a></p>
<form id=f><p><input type=file id=files multiple required></p>
<p><label>Sampling rate (Hz, needed for .npy/.csv) <input id=fs type=number value=500 style="width:6em"></label></p>
<p><label>API key (if enabled) <input id=key type=password></label></p><button>Predict</button></form>
<div id=out></div>
<script>
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const fd=new FormData();
for(const f of document.getElementById('files').files)fd.append('files',f);fd.append('fs',document.getElementById('fs').value);
const h={};const k=document.getElementById('key').value;if(k)h['X-API-Key']=k;
const r=await fetch('/predict',{method:'POST',body:fd,headers:h});const j=await r.json();const o=document.getElementById('out');
if(!r.ok){o.textContent=JSON.stringify(j.detail||j);return}
const rows=Object.entries(j.probabilities).sort((a,b)=>b[1]-a[1]).map(([l,p])=>
`<tr><td>${l}${j.positive.includes(l)?' ✓':''}</td><td style="width:60%"><div class=bar style="width:${(p*100).toFixed(1)}%"></div></td><td>${(p*100).toFixed(1)}%</td></tr>`).join('');
o.innerHTML=`<h3>Result (${j.n_windows} window(s), leads: ${j.active_leads.length})</h3><table>${rows}</table><p><small>${j.disclaimer}</small></p>`};
</script>"""


@public.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX_HTML


@public.get("/health")
def health():
    return {"status": "ok", "version": __version__, "model_loaded": state.predictor is not None,
            "auth_enabled": bool(C.API_KEY)}


# ------------------------------------------------------------------ secured: metadata
@secured.get("/labels")
def labels():
    return [{"name": l, "description": LABEL_DESCRIPTIONS.get(l, "")} for l in O.LABELS]


@secured.get("/sources/kinds")
def source_kinds():
    return {"kinds": list(KINDS),
            "notes": {"ptbxl": "root = folder with ptbxl_database.csv + scp_statements.csv",
                      "physionet_snomed": "root = folder with WFDB .hea/.mat files carrying '#Dx:' SNOMED codes "
                                          "(Chapman-Shaoxing/Ningbo, CinC 2020/2021 sets)",
                      "mimic_ecg": "root = MIMIC-IV-ECG folder (record_list.csv, machine_measurements.csv)",
                      "manifest": "generic CSV manifest; use for ACS-ECG, ZZU pECG, EchoNext, custom data",
                      "synthetic": "no root; options: n, seed, fs, annotated"}}


# ------------------------------------------------------------------ secured: dataset build
@secured.post("/datasets/build", status_code=202)
def build_dataset(req: BuildRequest):
    from .builder import build_store
    out = _store_dir(req.store_name)
    if (out / "info.json").exists() and not req.overwrite:
        raise HTTPException(409, f"store '{req.store_name}' already exists (set overwrite=true)")
    specs = []
    for s in req.sources:
        _name(s.name, "source name")
        specs.append((s.kind, s.name, _allowed_path(s.root) if s.root else None, s.options))

    def job(ctx: JobContext) -> dict:
        sources = []
        for kind, name, root, options in specs:        # constructed inside the job so errors are reported there
            ctx.log(f"opening source {name} ({kind})")
            sources.append(build_source(kind, name, root, options))
        info = build_store(sources, out, req.holdout_sources, req.max_records_per_source,
                           req.max_windows_per_record, req.overwrite, progress=ctx.progress, cancel=ctx.cancel)
        ctx.log(f"store ready: {info['n']} records {info['split_counts']} dropped={info['dropped']}")
        return {k: info[k] for k in ("n", "split_counts", "per_source", "dropped")} | {"store": req.store_name}

    j = state.jobs.submit("build_dataset", job, req.model_dump())
    return {"job_id": j.id, "status_url": f"/jobs/{j.id}"}


@secured.get("/datasets")
def list_datasets():
    out = []
    for d in sorted((C.WORKSPACE / "stores").glob("*/info.json")):
        info = json.loads(d.read_text())
        out.append({"name": d.parent.name, "n": info["n"], "sources": info["sources"],
                    "split_counts": info["split_counts"]})
    return out


@secured.get("/datasets/{name}")
def dataset_info(name: str):
    f = _store_dir(name) / "info.json"
    if not f.exists():
        raise HTTPException(404, "store not found")
    return json.loads(f.read_text())


# ------------------------------------------------------------------ secured: training
@secured.post("/train", status_code=202)
def start_training(req: TrainRequest):
    store, run = _store_dir(req.store_name), _run_dir(req.run_name)
    if not (store / "info.json").exists():
        raise HTTPException(404, f"store '{req.store_name}' not found; build it first")
    resume = None
    if req.resume_from_run:
        resume = _run_dir(req.resume_from_run) / "best.pt"
        if not resume.exists():
            raise HTTPException(404, f"run '{req.resume_from_run}' has no best.pt")
    cfg = TrainConfig(store_path=str(store), out_dir=str(run), resume_from=str(resume) if resume else None,
                      **req.hyper.model_dump())

    def job(ctx: JobContext) -> dict:
        from .train import train
        result = train(cfg, progress=ctx.progress, cancel=ctx.cancel, log=ctx.log)
        if req.activate:
            load_predictor(result["checkpoint"])
            ctx.log("model activated for /predict")
        return {"run": req.run_name, "best_epoch": result["best_epoch"],
                **{k: result[k]["macro_auroc"] for k in ("val", "test", "external") if k in result}}

    j = state.jobs.submit("train", job, req.model_dump())
    return {"job_id": j.id, "status_url": f"/jobs/{j.id}"}


@secured.get("/runs")
def list_runs():
    out = []
    for d in sorted((C.WORKSPACE / "runs").iterdir() if (C.WORKSPACE / "runs").exists() else []):
        if (d / "best.pt").exists():
            m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
            out.append({"name": d.name, "best_epoch": m.get("best_epoch"),
                        **{k: (m.get(k) or {}).get("macro_auroc") for k in ("val", "test", "external")}})
    return out


@secured.get("/runs/{name}")
def run_metrics(name: str):
    f = _run_dir(name) / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "run not found or not finished")
    return json.loads(f.read_text())


# ------------------------------------------------------------------ secured: jobs
@secured.get("/jobs")
def list_jobs():
    return [j.to_dict(logs=False) for j in state.jobs.list()]


@secured.get("/jobs/{job_id}")
def get_job(job_id: str, tail: int = 100):
    j = state.jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j.to_dict(tail=max(0, min(tail, 1000)))


@secured.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not state.jobs.cancel(job_id):
        raise HTTPException(409, "job not found or already finished")
    return {"job_id": job_id, "cancel_requested": True}


# ------------------------------------------------------------------ secured: model + prediction
@secured.get("/model")
def model_info():
    if state.predictor is None:
        raise HTTPException(404, "no model loaded; train one or POST /model/load")
    return state.predictor.info()


@secured.post("/model/load")
def model_load(req: LoadModelRequest):
    p = _run_dir(req.run_name) / "best.pt"
    if not p.exists():
        raise HTTPException(404, "run has no best.pt")
    try:
        return load_predictor(str(p))
    except Exception as e:                              # noqa: BLE001
        raise HTTPException(422, f"could not load checkpoint: {e}") from e


def _need_model():
    if state.predictor is None:
        raise HTTPException(503, "no model loaded; train one (POST /train) or POST /model/load")
    return state.predictor


def _oriented(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(422, "signal must be 2-D")
    return arr.T if arr.shape[0] > arr.shape[1] else arr          # leads = the shorter axis


def _parse_upload(files: list[tuple[str, bytes]], fs: float | None) -> tuple[np.ndarray, float, list[str] | None, float]:
    """-> (signal (leads, samples), fs, lead_names, unit_scale)"""
    by_ext = {Path(n).suffix.lower(): (n, b) for n, b in files}
    if ".hea" in by_ext:                                           # WFDB pair
        import wfdb
        from .sources.base import unit_scale_from
        with tempfile.TemporaryDirectory() as td:
            for n, b in files:
                (Path(td) / Path(n).name).write_bytes(b)           # basename only: no path traversal
            rec = Path(by_ext[".hea"][0]).stem
            try:
                sig, fields = wfdb.rdsamp(str(Path(td) / rec))
            except Exception as e:                                 # noqa: BLE001
                raise HTTPException(422, f"could not read WFDB record (upload both .hea and .dat): {e}") from e
        return sig.T, float(fields["fs"]), list(fields["sig_name"]), unit_scale_from((fields.get("units") or ["mV"])[0])
    if fs is None:
        raise HTTPException(422, "form field 'fs' (sampling rate in Hz) is required for .npy/.csv uploads")
    if ".npy" in by_ext:
        try:
            arr = np.load(io.BytesIO(by_ext[".npy"][1]), allow_pickle=False)
        except Exception as e:                                     # noqa: BLE001
            raise HTTPException(422, f"invalid .npy file: {e}") from e
        return _oriented(arr), fs, None, 1.0
    if ".csv" in by_ext:
        import pandas as pd
        from .signal_utils import normalize_lead_name
        raw = by_ext[".csv"][1]
        try:
            df = pd.read_csv(io.BytesIO(raw))
            known = [c for c in df.columns if normalize_lead_name(str(c))]
            if known:                                              # header row with lead names
                return df[known].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).T, fs, \
                    [str(c) for c in known], 1.0
            df = pd.read_csv(io.BytesIO(raw), header=None)         # headerless: assume standard lead order
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0)
            return _oriented(df.to_numpy(np.float32)), fs, None, 1.0
        except HTTPException:
            raise
        except Exception as e:                                     # noqa: BLE001
            raise HTTPException(422, f"invalid .csv file: {e}") from e
    raise HTTPException(415, "unsupported upload; send .npy, .csv, or a WFDB .hea + .dat pair")


@secured.post("/predict")
async def predict(files: list[UploadFile] = File(...), fs: float | None = Form(None),
                  max_windows: int = Form(6)):
    predictor = _need_model()
    payload: list[tuple[str, bytes]] = []
    total = 0
    for f in files:
        data = await f.read(C.MAX_UPLOAD_BYTES + 1)
        total += len(data)
        if total > C.MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload larger than {C.MAX_UPLOAD_BYTES // 2**20} MB")
        payload.append((f.filename or "upload", data))
    sig, rate, names, scale = await run_in_threadpool(_parse_upload, payload, fs)
    try:
        res = await run_in_threadpool(predictor.predict, sig, rate, names, scale, max(1, min(max_windows, 30)))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return res | {"disclaimer": C.DISCLAIMER}


@secured.post("/predict/json")
async def predict_json(req: PredictJSON):
    predictor = _need_model()
    arr = np.asarray(req.signal, dtype=np.float32)
    if arr.ndim != 2 or arr.size > 12 * 10000 * 60:
        raise HTTPException(422, "signal must be a 2-D array (max 60 s at 10 kHz x 12 leads)")
    if req.orientation == "samples_first":
        arr = arr.T
    try:
        res = await run_in_threadpool(predictor.predict, arr, req.fs, req.lead_names,
                                      1e-3 if req.unit == "uV" else 1.0)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return res | {"disclaimer": C.DISCLAIMER}


app.include_router(public)
app.include_router(secured)
