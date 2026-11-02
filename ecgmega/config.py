"""Global constants and environment-driven settings."""
from __future__ import annotations

import os
from pathlib import Path

# ---- signal standard -------------------------------------------------------
STORE_FS = 500            # sampling rate of the unified store (Hz)
WINDOW_SEC = 10           # every stored record is a 10 s window
WINDOW_LEN = STORE_FS * WINDOW_SEC   # 5000 samples
N_LEADS = 12
STD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

BAND_LOW_HZ = 0.5
BAND_HIGH_HZ = 40.0

# ---- split policy ----------------------------------------------------------
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_EXTERNAL = 0, 1, 2, 3
SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "external"}

# ---- service settings ------------------------------------------------------
WORKSPACE = Path(os.environ.get("ECG_WORKSPACE", "./workspace")).resolve()
# The API only touches server paths under these roots (os.pathsep separated).
ALLOWED_ROOTS = [Path(p).resolve() for p in
                 os.environ.get("ECG_ALLOWED_ROOTS", str(WORKSPACE)).split(os.pathsep) if p]
API_KEY = os.environ.get("ECG_API_KEY", "")          # empty = auth disabled
MAX_UPLOAD_BYTES = int(os.environ.get("ECG_MAX_UPLOAD_MB", "20")) * 1024 * 1024
DEFAULT_CHECKPOINT = os.environ.get("ECG_CHECKPOINT", "")

DISCLAIMER = ("Research use only. Not a medical device; outputs are not a diagnosis "
              "and must not be used for clinical decisions.")
