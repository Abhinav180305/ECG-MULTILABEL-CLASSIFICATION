# ECG Mega (v2)

Merge many public ECG datasets into **one unified store**, train **one multi-label model** on it, and
serve it with **FastAPI**. Research use only; not a medical device.

```
sources (PTB-XL, Chapman/Ningbo, CinC, MIMIC-IV-ECG, ACS-ECG, ZZU pECG, EchoNext, your own)
   -> harmonize (12 std leads, mV, 500 Hz, 0.5-40 Hz band-pass, 10 s windows)
   -> quality filter -> de-duplicate (signal hash) -> unified labels + label mask
   -> patient-level split (train/val/test, whole sources can be held out as "external")
   -> store/  (numpy memmap: signals.bin float16 + labels/mask/lead_mask/split .npy + meta.csv)
   -> train (optional masked-reconstruction pretraining -> masked-BCE fine-tuning, source-balanced sampling)
   -> best.pt (weights + label list + per-label thresholds + metrics)
   -> FastAPI: /datasets/build, /train, /jobs, /predict, ...
```

## Quick start (synthetic data, no downloads)

```bash
pip install -r requirements.txt
export ECG_WORKSPACE=./workspace ECG_API_KEY=change-me      # key optional; omit to disable auth
uvicorn ecgmega.api:app --port 8000                          # docs at http://localhost:8000/docs

H='-H X-API-Key:change-me -H Content-Type:application/json'
curl $H localhost:8000/datasets/build -d '{"store_name":"demo","holdout_sources":["b"],"sources":[
  {"name":"a","kind":"synthetic","options":{"n":600,"seed":1}},
  {"name":"b","kind":"synthetic","options":{"n":150,"seed":2,"fs":250}}]}'
curl $H localhost:8000/train -d '{"store_name":"demo","run_name":"r1","hyper":{"model_size":"tiny","epochs":10}}'
curl -H X-API-Key:change-me localhost:8000/jobs            # progress, logs, metrics
curl -H X-API-Key:change-me -F files=@ecg.npy -F fs=500 localhost:8000/predict
```
Open `http://localhost:8000/` for a small upload page.

## Real datasets

Put datasets under a folder the server may read: `export ECG_ALLOWED_ROOTS=/data` (the API refuses other
paths). `configs/sources.example.json` shows every source kind:

| kind | use for | notes |
|---|---|---|
| `ptbxl` | PTB-XL | needs `ptbxl_database.csv`, `scp_statements.csv` |
| `physionet_snomed` | Chapman-Shaoxing/Ningbo, CinC 2020/21 sets | reads `#Dx:` SNOMED codes from `.hea`; no patient ids exist, so patient-level leakage cannot be excluded |
| `mimic_ecg` | MIMIC-IV-ECG (+ MEETI/glucose are derived from it) | labels from machine-report keywords: noisier than cardiologist labels |
| `manifest` | ACS-ECG, ZZU pECG, EchoNext, Brugada-HUCA, your data | CSV with `path`, `labels`, ...; `label_map`, `annotated`, `flag_columns`, `format: wfdb\|npy` |
| `synthetic` | smoke tests | |

**I could not download or inspect ACS-ECG, ZZU pECG, EchoNext or Harvard-Emory from the build sandbox**, so these
use the generic `manifest` adapter: convert/describe them with a `manifest.csv` and a `label_map`. Check each
dataset's license/access terms before merging.

CLI without the API: `python scripts/build_dataset.py configs/sources.example.json workspace/stores/mega --holdout acs_ecg`
then `python scripts/train.py workspace/stores/mega workspace/runs/r1 --size base --epochs 30 --pretrain-epochs 5`.

## Design decisions that matter for a merged dataset

* **Label mask.** Every source declares which labels it annotates (`annotated`). Unannotated labels get mask 0
  and contribute **no loss**, instead of being treated as negatives. Evaluation uses the same mask.
* **Hierarchy.** LBBB / RBBB / IAVB / LAFB imply CD.
* **De-duplication** by content hash across sources (PTB-XL is inside the CinC bundles; MEETI etc. derive from
  MIMIC-IV-ECG). Best-effort: it only catches records that harmonize identically.
* **Splits** are deterministic and patient-level (`md5(source:patient) % 10`: 0-7 train, 8 val, 9 test). Sources
  in `holdout_sources` are never trained on and reported as `external`.
* **Source-balanced sampling** (`source_alpha`, default 0.5) so MIMIC (~800k) does not swamp small sets.
* **Metrics** are masked macro-AUROC overall, per label and per source; thresholds are tuned for F1 on val.
* **Security.** Server paths must be inside `ECG_ALLOWED_ROOTS`; manifest records cannot escape their root;
  `.npy` is loaded with `allow_pickle=False`; checkpoints with `torch.load(weights_only=True)`; uploads are size
  capped; optional `X-API-Key` on everything except `/` and `/health`.

## API

| method + path | purpose |
|---|---|
| `GET /health`, `GET /` | liveness, upload page (public) |
| `GET /labels`, `GET /sources/kinds` | ontology, source kinds |
| `POST /datasets/build` -> job | build a store; `GET /datasets`, `GET /datasets/{name}` for stats |
| `POST /train` -> job | train on a store (`hyper`: size, epochs, pretrain_epochs, lr, ...); activates the model when done |
| `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` | progress, logs, results, cooperative cancel |
| `GET /runs`, `GET /runs/{name}` | finished runs and metrics |
| `GET /model`, `POST /model/load` | current model info / switch run |
| `POST /predict` | upload `.npy`, `.csv`, or WFDB `.hea`+`.dat` (+ `fs` for npy/csv) |
| `POST /predict/json` | `{signal, fs, lead_names?, orientation?, unit?}` |

## Tests

`pytest -q` runs everything installed: `tests/test_data.py` (numpy/scipy/pandas only), `tests/test_model_train.py`
(needs torch), `tests/test_api.py` (needs torch + fastapi + httpx; runs build -> train -> predict over HTTP).

## Status of this code

The data layer (harmonization, sources, store builder, dedup, masks, splits, job manager) was executed and its
tests pass. **The PyTorch model/training/inference code and the FastAPI app were written but never executed** (the
authoring sandbox could not install torch or fastapi). Run `pytest -q` first; fix anything it surfaces.
Also expect to tune: model size, epochs, label maps for each real dataset, and class thresholds.
