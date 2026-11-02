"""Test the running API on real PTB-XL records the model never trained on.

Run from the project folder (server must be running with a model loaded):
    python test_records.py
Needs testdata/ptbxl_database.csv (see the download line in the instructions).
Set API_KEY below only if you started the server with ECG_API_KEY.
"""
import ast
import hashlib
import io
import sys

import httpx
import pandas as pd

API = "http://localhost:8000"
API_KEY = ""                      # e.g. "change-me" if you set ECG_API_KEY
BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
CODES = ["NORM", "AFIB", "CLBBB", "SBRAD", "STACH", "IMI"]   # what to try
PER_CODE = 2

headers = {"X-API-Key": API_KEY} if API_KEY else {}
df = pd.read_csv("testdata/ptbxl_database.csv", index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)


def is_test_patient(pid) -> bool:
    """Same rule the dataset builder uses: md5('ptbxl:<patient>') % 10 == 9 -> held-out test split."""
    return int(hashlib.md5(f"ptbxl:{int(pid)}".encode()).hexdigest()[:8], 16) % 10 == 9


def pick(code):
    out = []
    for ecg_id, row in df.iterrows():
        if code in row["scp_codes"] and is_test_patient(row["patient_id"]):
            out.append(ecg_id)
            if len(out) == PER_CODE:
                break
    return out


with httpx.Client(timeout=120, follow_redirects=True) as c:
    if c.get(f"{API}/model", headers=headers).status_code != 200:
        sys.exit("No model loaded: POST /model/load first (see /docs).")
    for code in CODES:
        for ecg_id in pick(code):
            stem = df.loc[ecg_id, "filename_hr"]               # e.g. records500/00000/00001_hr
            name = stem.split("/")[-1]
            files = []
            for ext in (".hea", ".dat"):
                r = c.get(BASE + stem + ext)
                r.raise_for_status()
                files.append(("files", (name + ext, r.content)))
            res = c.post(f"{API}/predict", files=files, headers=headers)
            if res.status_code != 200:
                print(ecg_id, "ERROR", res.status_code, res.text[:200])
                continue
            j = res.json()
            top = sorted(j["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            print(f"ecg_id {ecg_id}: true={df.loc[ecg_id, 'scp_codes']}")
            print("   model says:", j["positive"], "| top3:", [(k, round(v, 2)) for k, v in top])
