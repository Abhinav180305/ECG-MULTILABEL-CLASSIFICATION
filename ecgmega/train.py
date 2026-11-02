"""Training: optional self-supervised pretraining, then masked multi-label fine-tuning on the unified store."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from . import config as C
from . import ontology as O
from .dataset import ECGDataset
from .errors import Cancelled
from .model import build_model
from .store import UnifiedStore


class TrainHyper(BaseModel):
    """Hyper-parameters (shared by the CLI and the API)."""
    model_size: Literal["tiny", "small", "base", "large"] = "base"
    epochs: int = Field(30, ge=1)
    pretrain_epochs: int = Field(0, ge=0, description="self-supervised masked-reconstruction epochs first")
    batch_size: int = Field(64, ge=2)
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.3
    train_fs: int = 100
    crop_sec: float = 8.0
    source_alpha: float = Field(0.5, ge=0.0, le=1.0, description="0 = equal weight per source, 1 = uniform")
    pos_weight: bool = True
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    patience: int = 8
    mask_ratio: float = 0.3
    max_train_samples: int | None = None
    seed: int = 0


class TrainConfig(TrainHyper):
    store_path: str
    out_dir: str
    resume_from: str | None = Field(None, description="checkpoint to initialise weights from")


# ------------------------------------------------------------------ losses
def masked_bce(logits, y, m, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def compute_pos_weight(store: UnifiedStore, idx: np.ndarray, clip: float = 20.0) -> torch.Tensor:
    y, m = store.labels[idx].astype(np.float64), store.mask[idx].astype(np.float64)
    pos = (y * m).sum(0)
    neg = ((1 - y) * m).sum(0)
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, clip), dtype=torch.float32)


def time_mask(x: torch.Tensor, ratio: float, patch: int = 25) -> torch.Tensor:
    """(B,1,T) float mask, 1 where the signal is hidden (random contiguous patches)."""
    b, _, t = x.shape
    n = -(-t // patch)
    m = (torch.rand(b, n, device=x.device) < ratio).float().repeat_interleave(patch, dim=1)[:, :t]
    return m.unsqueeze(1)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    P, Y, M, S = [], [], [], []
    for b in loader:
        logits = model(b["x"].to(device))
        P.append(torch.sigmoid(logits).float().cpu().numpy())
        Y.append(b["y"].numpy())
        M.append(b["m"].numpy())
        S.append(np.asarray(b["src"]))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(M), np.concatenate(S)


def _macro_auroc(p, y, m) -> tuple[float | None, dict[str, float]]:
    per = {}
    for j, lab in enumerate(O.LABELS):
        ok = m[:, j] > 0
        yj = y[ok, j]
        if ok.sum() < 2 or yj.min() == yj.max():
            continue
        per[lab] = float(roc_auc_score(yj, p[ok, j]))
    return (float(np.mean(list(per.values()))) if per else None), per


def compute_metrics(p, y, m, src, source_names: list[str]) -> dict:
    macro, per = _macro_auroc(p, y, m)
    per_source = {}
    for si, name in enumerate(source_names):
        sel = src == si
        if sel.sum() > 1:
            s_macro, _ = _macro_auroc(p[sel], y[sel], m[sel])
            per_source[name] = s_macro
    return {"n": int(len(p)), "macro_auroc": macro, "per_label_auroc": per, "per_source_macro_auroc": per_source}


def best_thresholds(p, y, m) -> list[float]:
    grid = np.arange(0.05, 0.96, 0.05)
    out = []
    for j in range(p.shape[1]):
        ok = m[:, j] > 0
        if ok.sum() < 2 or y[ok, j].sum() == 0:
            out.append(0.5)
            continue
        yj, pj = y[ok, j], p[ok, j]
        f1s = []
        for t in grid:
            pred = pj >= t
            tp = float((pred & (yj > 0)).sum())
            f1s.append(2 * tp / max(1.0, pred.sum() + yj.sum()))
        out.append(float(grid[int(np.argmax(f1s))]))
    return out


def _plain(obj):
    return json.loads(json.dumps(obj, default=float))


# ------------------------------------------------------------------ main entry
def train(cfg: TrainConfig, progress: Callable[[float, str], None] | None = None,
          cancel: threading.Event | None = None, log: Callable[[str], None] = print) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                       "cpu" if cfg.device == "auto" else cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))

    store = UnifiedStore(cfg.store_path)
    if list(store.info["labels"]) != O.LABELS:
        raise ValueError("the store was built with a different label list than the current ontology")

    mk = lambda splits, aug, mx=None: ECGDataset(store, splits, cfg.train_fs, aug, cfg.crop_sec, mx, cfg.seed)
    ds_tr = mk([C.SPLIT_TRAIN], True, cfg.max_train_samples)
    ds_va, ds_te, ds_ex = mk([C.SPLIT_VAL], False), mk([C.SPLIT_TEST], False), mk([C.SPLIT_EXTERNAL], False)
    if len(ds_tr) < 2:
        raise ValueError("training split has fewer than 2 records")
    log(f"device={dev} train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} external={len(ds_ex)}")

    pin = dev.type == "cuda"
    nw = cfg.num_workers
    common = dict(num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    dl_tr = DataLoader(ds_tr, cfg.batch_size, sampler=ds_tr.make_sampler(cfg.source_alpha, cfg.seed),
                       drop_last=len(ds_tr) > cfg.batch_size, **common)
    mk_eval = lambda ds: DataLoader(ds, cfg.batch_size, shuffle=False, **common) if len(ds) else None
    dl_va, dl_te, dl_ex = mk_eval(ds_va), mk_eval(ds_te), mk_eval(ds_ex)

    model = build_model(O.N_LABELS, cfg.model_size, cfg.dropout).to(dev)
    if cfg.resume_from:
        ck = torch.load(cfg.resume_from, map_location="cpu", weights_only=True)
        own = model.state_dict()
        keep = {k: v for k, v in ck["state_dict"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        log(f"initialised {len(keep)}/{len(own)} tensors from {cfg.resume_from}")

    total_epochs = cfg.pretrain_epochs + cfg.epochs
    done_epochs = 0
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled("training cancelled")

    # ---- stage A: self-supervised masked reconstruction
    if cfg.pretrain_epochs:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for ep in range(cfg.pretrain_epochs):
            model.train()
            tot, cnt = 0.0, 0
            for b in dl_tr:
                check_cancel()
                x = b["x"].to(dev)
                hide = time_mask(x, cfg.mask_ratio)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    rec = model.reconstruct(x * (1 - hide))
                    loss = (((rec - x) ** 2) * hide).sum() / (hide.sum() * x.shape[1]).clamp(min=1.0)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot, cnt = tot + float(loss), cnt + 1
            done_epochs += 1
            log(f"[pretrain {ep + 1}/{cfg.pretrain_epochs}] recon_loss={tot / max(cnt, 1):.4f}")
            if progress:
                progress(done_epochs / total_epochs, f"pretrain epoch {ep + 1}")

    # ---- stage B: masked multi-label fine-tuning
    pw = compute_pos_weight(store, ds_tr.idx).to(dev) if cfg.pos_weight else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl_tr), pct_start=0.2)
    best, best_ep, bad = -1e9, -1, 0
    history: list[dict] = []
    ckpt_path = out / "best.pt"

    def save(extra: dict):
        torch.save(_plain({"labels": O.LABELS, "model_cfg": model.cfg, "train_fs": cfg.train_fs,
                           "store_fs": C.STORE_FS, "source_names": store.source_names, **extra})
                   | {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, ckpt_path)

    for ep in range(cfg.epochs):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for b in dl_tr:
            check_cancel()
            x, y, m = b["x"].to(dev), b["y"].to(dev), b["m"].to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = masked_bce(model(x).float(), y, m, pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, cnt = tot + float(loss), cnt + 1
        tr_loss = tot / max(cnt, 1)
        rec = {"epoch": ep + 1, "train_loss": tr_loss, "seconds": round(time.time() - t0, 1)}
        if dl_va is not None:
            p, y_, m_, s_ = predict_loader(model, dl_va, dev)
            vm = compute_metrics(p, y_, m_, s_, store.source_names)
            rec["val_macro_auroc"] = vm["macro_auroc"]
            score = vm["macro_auroc"] if vm["macro_auroc"] is not None else -tr_loss
        else:
            score = -tr_loss
        history.append(rec)
        done_epochs += 1
        log(f"[epoch {ep + 1}/{cfg.epochs}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in rec.items() if k != "epoch"))
        if progress:
            progress(done_epochs / total_epochs, f"epoch {ep + 1}/{cfg.epochs}")
        if score > best + 1e-6:
            best, best_ep, bad = score, ep + 1, 0
            save({"best_epoch": best_ep})
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stopping at epoch {ep + 1} (best epoch {best_ep})")
                break

    # ---- final evaluation with the best weights
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["state_dict"])
    result: dict = {"best_epoch": best_ep, "history": history, "checkpoint": str(ckpt_path)}
    thresholds = [0.5] * O.N_LABELS
    if dl_va is not None:
        p, y_, m_, _ = predict_loader(model, dl_va, dev)
        thresholds = best_thresholds(p, y_, m_)
    for name, dl in (("val", dl_va), ("test", dl_te), ("external", dl_ex)):
        if dl is not None:
            p, y_, m_, s_ = predict_loader(model, dl, dev)
            result[name] = compute_metrics(p, y_, m_, s_, store.source_names)
            log(f"{name}: macro AUROC = {result[name]['macro_auroc']}")
    save({"best_epoch": best_ep, "thresholds": thresholds,
          "metrics": {k: result[k] for k in ("val", "test", "external") if k in result}})
    (out / "metrics.json").write_text(json.dumps(_plain({k: v for k, v in result.items()}), indent=2))
    if progress:
        progress(1.0, "finished")
    return _plain(result)
