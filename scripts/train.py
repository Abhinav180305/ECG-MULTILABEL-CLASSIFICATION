#!/usr/bin/env python
"""Train from the command line (no API needed).

    python scripts/train.py workspace/stores/mega workspace/runs/r1 --size base --epochs 30 --pretrain-epochs 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgmega.train import TrainConfig, train     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("out")
    ap.add_argument("--size", default="base", choices=["tiny", "small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume-from", default=None)
    a = ap.parse_args()
    cfg = TrainConfig(store_path=a.store, out_dir=a.out, model_size=a.size, epochs=a.epochs,
                      pretrain_epochs=a.pretrain_epochs, batch_size=a.batch_size, lr=a.lr,
                      num_workers=a.workers, device=a.device, resume_from=a.resume_from)
    res = train(cfg, progress=lambda p, m: None)
    print("checkpoint:", res["checkpoint"])
    for k in ("val", "test", "external"):
        if k in res:
            print(k, "macro AUROC:", res[k]["macro_auroc"])


if __name__ == "__main__":
    main()
