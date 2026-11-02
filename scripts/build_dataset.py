#!/usr/bin/env python
"""Build the unified store from a JSON spec (no API needed).

    python scripts/build_dataset.py configs/sources.example.json workspace/stores/mega --holdout acs_ecg
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgmega.builder import build_store          # noqa: E402
from ecgmega.sources import build_source         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="JSON list of {name, kind, root, options}")
    ap.add_argument("out", help="output store directory")
    ap.add_argument("--holdout", nargs="*", default=[], help="source names kept as external test sets")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=1, help="10 s windows per long recording")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    specs = json.loads(Path(a.spec).read_text())
    sources = [build_source(s["kind"], s["name"], s.get("root"), s.get("options")) for s in specs]
    info = build_store(sources, a.out, a.holdout, a.max_per_source, a.max_windows, a.overwrite,
                       progress=lambda p, m: print(f"\r{p * 100:5.1f}%  {m:<60}", end="", flush=True))
    print()
    print(json.dumps({k: info[k] for k in ("n", "split_counts", "per_source", "dropped")}, indent=2))


if __name__ == "__main__":
    main()
