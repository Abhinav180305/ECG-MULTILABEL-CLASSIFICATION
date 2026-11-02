"""Load a trained checkpoint and score arbitrary 12-lead recordings."""
from __future__ import annotations
 
import threading
from pathlib import Path
from typing import Sequence
 
import numpy as np
import torch
 
from . import config as C
from . import ontology as O
from . import signal_utils as S
from .model import build_model
 
 
class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.path = str(checkpoint)
        # weights_only=True: refuses to unpickle arbitrary objects, so a malicious .pt cannot run code
        ck = torch.load(self.path, map_location="cpu", weights_only=True)
        self.labels: list[str] = list(ck["labels"])
        if len(self.labels) != O.N_LABELS:
            raise ValueError("checkpoint label list does not match the current ontology")
        self.thresholds = list(ck.get("thresholds") or [0.5] * len(self.labels))
        self.train_fs = int(ck["train_fs"])
        self.step = C.STORE_FS // self.train_fs
        mc = ck["model_cfg"]
        self.model = build_model(mc["n_labels"], mc["size"], mc.get("dropout", 0.3))
        self.model.load_state_dict(ck["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.metrics = ck.get("metrics", {})
        self.active = O.active_labels(self.labels, ck.get("trained_labels"), self.metrics)
        self._active_set = set(self.active)
        self.best_epoch = ck.get("best_epoch")
        self._lock = threading.Lock()
 
    def info(self) -> dict:
        return {"checkpoint": self.path, "labels": self.labels, "active_labels": self.active,
                "thresholds": dict(zip(self.labels, self.thresholds)),
                "train_fs": self.train_fs, "model": self.model.cfg, "best_epoch": self.best_epoch,
                "metrics": {k: v.get("macro_auroc") for k, v in self.metrics.items() if isinstance(v, dict)}}
 
    def predict(self, signal: np.ndarray, fs: float, lead_names: Sequence[str] | None = None,
                unit_scale: float = 1.0, max_windows: int = 6) -> dict:
        """signal: (n_leads, n_samples) in millivolts (or scaled by unit_scale). Lead names default to
        the standard 12-lead order when the array has exactly 12 rows."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2:
            raise ValueError("signal must be 2-D (leads x samples)")
        names = list(lead_names) if lead_names else (list(C.STD_LEADS) if signal.shape[0] == C.N_LEADS else None)
        if names is None or len(names) != signal.shape[0]:
            raise ValueError("provide lead_names (one per row), or a 12-row signal in standard lead order")
        sig, lead_mask = S.harmonize(signal, fs, names, unit_scale)
        if lead_mask.sum() < 2:
            raise ValueError("fewer than 2 recognised leads")
        wins = S.to_windows(sig, max_windows=max_windows)
        if not wins:
            raise ValueError("recording too short (need at least 8 s)")
        wins = [w for w in wins if S.passes_quality(w, lead_mask)]
        if not wins:
            raise ValueError("signal quality too low (flat, non-finite or out-of-range)")
        x = np.stack([S.zscore(w[:, ::self.step]) * lead_mask[:, None] for w in wins]).astype(np.float32)
        with self._lock, torch.inference_mode():
            probs = torch.sigmoid(self.model(torch.from_numpy(x).to(self.device))).mean(0).cpu().numpy()
        res = {lab: float(p) for lab, p in zip(self.labels, probs) if lab in self._active_set}
        return {
            "probabilities": res,
            "positive": [l for l, p, t in zip(self.labels, probs, self.thresholds)
                         if l in self._active_set and p >= t],
            "n_windows": len(wins),
            "active_leads": [n for n, m in zip(C.STD_LEADS, lead_mask) if m],
            "duration_sec": round(signal.shape[1] / fs, 2),
        }
 
