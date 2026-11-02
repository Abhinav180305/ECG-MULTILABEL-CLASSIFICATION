"""Source registry: build_source(kind, root, **options) -> BaseSource."""
from __future__ import annotations

from .base import BaseSource, Record

KINDS = ("ptbxl", "physionet_snomed", "mimic_ecg", "manifest", "synthetic")


def build_source(kind: str, name: str, root: str | None = None, options: dict | None = None) -> BaseSource:
    options = dict(options or {})
    if kind == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(name=name, root=root, **options)
    if root is None:
        raise ValueError(f"source '{name}' of kind '{kind}' needs a root path")
    if kind == "ptbxl":
        from .ptbxl import PTBXLSource
        return PTBXLSource(root, name=name, **options)
    if kind == "physionet_snomed":
        from .physionet_snomed import PhysioNetSnomedSource
        return PhysioNetSnomedSource(root, name=name, **options)
    if kind == "mimic_ecg":
        from .mimic_ecg import MimicECGSource
        return MimicECGSource(root, name=name, **options)
    if kind == "manifest":
        from .manifest import ManifestSource
        return ManifestSource(root, name=name, **options)
    raise ValueError(f"unknown source kind '{kind}'; choose from {KINDS}")


__all__ = ["BaseSource", "Record", "build_source", "KINDS"]
