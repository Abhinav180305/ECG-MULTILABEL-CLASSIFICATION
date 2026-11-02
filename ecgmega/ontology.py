"""Unified label ontology plus per-source code maps.
 
Each source maps its own label system (SCP-ECG, SNOMED CT, free-text reports, ...) onto
LABELS. Sources also declare which labels they *annotate*; labels a source does not annotate
get mask=0 so a missing label is never treated as a negative.
"""
from __future__ import annotations
 
import re
from typing import Iterable
 
LABELS = [
    "NORM",   # normal / sinus rhythm without abnormality
    "MI",     # myocardial infarction
    "STTC",   # ST/T change
    "CD",     # conduction disturbance (superclass; implied by LBBB/RBBB/IAVB/LAFB)
    "HYP",    # hypertrophy
    "AFIB",   # atrial fibrillation
    "AFLT",   # atrial flutter
    "SBRAD",  # sinus bradycardia
    "STACH",  # sinus tachycardia
    "SARRH",  # sinus arrhythmia
    "LBBB",   # left bundle branch block
    "RBBB",   # right bundle branch block
    "IAVB",   # first-degree AV block
    "PVC",    # premature ventricular contraction
    "PAC",    # premature atrial / supraventricular contraction
    "LAFB",   # left anterior fascicular block
    "PACE",   # paced rhythm
    "SHD",    # structural heart disease (echo-derived; EchoNext style)
]
LABEL_INDEX = {l: i for i, l in enumerate(LABELS)}
N_LABELS = len(LABELS)
 
# child -> parent implied labels
HIERARCHY = {"LBBB": "CD", "RBBB": "CD", "IAVB": "CD", "LAFB": "CD"}
 
# Label groups that conventional source families annotate.
RHYTHM = ["AFIB", "AFLT", "SBRAD", "STACH", "SARRH", "PVC", "PAC", "PACE"]
CONDUCTION = ["CD", "LBBB", "RBBB", "IAVB", "LAFB"]
DIAGNOSTIC = ["NORM", "MI", "STTC", "CD", "HYP"]
ALL_BUT_SHD = [l for l in LABELS if l != "SHD"]
 
 
def expand_hierarchy(labels: Iterable[str]) -> list[str]:
    out = set(labels)
    for child, parent in HIERARCHY.items():
        if child in out:
            out.add(parent)
    return sorted(out, key=LABEL_INDEX.get)
 
 
def to_multihot(labels: Iterable[str]) -> list[int]:
    v = [0] * N_LABELS
    for l in expand_hierarchy(l for l in labels if l in LABEL_INDEX):
        v[LABEL_INDEX[l]] = 1
    return v
 
 
def to_mask(annotated: Iterable[str]) -> list[int]:
    ann = set(annotated)
    for child, parent in HIERARCHY.items():      # annotating a child implies annotating its parent
        if child in ann:
            ann.add(parent)
    return [1 if l in ann else 0 for l in LABELS]
 
 
# ---------------------------------------------------------------- PTB-XL (SCP-ECG)
PTBXL_CODE_MAP = {
    "NORM": "NORM", "AFIB": "AFIB", "AFLT": "AFLT", "SBRAD": "SBRAD", "STACH": "STACH",
    "SARRH": "SARRH", "CLBBB": "LBBB", "ILBBB": "LBBB", "CRBBB": "RBBB", "IRBBB": "RBBB",
    "1AVB": "IAVB", "PVC": "PVC", "PAC": "PAC", "LAFB": "LAFB", "PACE": "PACE",
}
# scp_statements.csv diagnostic_class values that are already in the ontology
PTBXL_SUPERCLASS = {"NORM", "MI", "STTC", "CD", "HYP"}
 
# ---------------------------------------------------------------- PhysioNet/CinC + Chapman (SNOMED CT)
SNOMED_MAP = {
    "426783006": "NORM",   # sinus rhythm (the Challenge treats this as the normal class)
    "164889003": "AFIB", "164890007": "AFLT",
    "426177001": "SBRAD", "427084000": "STACH", "427393009": "SARRH",
    "164909002": "LBBB", "733534002": "LBBB",
    "59118001": "RBBB", "713427006": "RBBB", "713426002": "RBBB",
    "270492004": "IAVB",
    "284470004": "PAC", "63593006": "PAC", "17338001": "PVC",
    "427172004": "PVC", "164884008": "PVC",
    "164865005": "MI", "164861001": "MI",
    "164934002": "STTC", "59931005": "STTC", "429622005": "STTC", "164931005": "STTC",
    "164873001": "HYP", "55827005": "HYP",
    "445118002": "LAFB",
    "10370003": "PACE", "251266004": "PACE",
}
# Rhythm/conduction oriented databases (Chapman-Shaoxing, Ningbo, CPSC, Georgia...) do not
# reliably annotate MI / hypertrophy, so those are masked out by default.
SNOMED_DEFAULT_ANNOTATED = ["NORM"] + RHYTHM + CONDUCTION + ["STTC"]
 
# ---------------------------------------------------------------- free-text machine reports (MIMIC-IV-ECG, MEETI)
_TEXT_RULES: list[tuple[str, str]] = [
    ("NORM", r"(?<![a-z])normal ecg"),
    ("AFIB", r"atrial fibrillation"),
    ("AFLT", r"atrial flutter"),
    ("SBRAD", r"sinus bradycardia"),
    ("STACH", r"sinus tachycardia"),
    ("SARRH", r"sinus arrhythmia"),
    ("LBBB", r"left bundle branch block"),
    ("RBBB", r"right bundle branch block"),
    ("IAVB", r"(first|1st) degree (a-?v|atrioventricular) block"),
    ("PVC", r"ventricular premature|premature ventricular|\bpvcs?\b"),
    ("PAC", r"(atrial|supraventricular) premature|premature (atrial|supraventricular)"),
    ("LAFB", r"left anterior fascicular|left anterior hemiblock"),
    ("PACE", r"\bpaced\b|ventricular.paced|atrial.paced|pacemaker"),
    ("MI", r"infarct"),
    ("STTC", r"st[- ]?t|t wave|st depression|st elevation|repolari[sz]ation|ischemi"),
    ("HYP", r"hypertrophy|enlargement"),
    ("CD", r"conduction (delay|defect)|bundle branch block|fascicular block"),
]
TEXT_RULES = [(lab, re.compile(pat)) for lab, pat in _TEXT_RULES]
TEXT_DEFAULT_ANNOTATED = ALL_BUT_SHD
 
 
def labels_from_text(text: str) -> list[str]:
    t = text.lower()
    found = [lab for lab, rx in TEXT_RULES if rx.search(t)]
    # a "normal ECG" statement contradicts any abnormal finding
    if "NORM" in found and len(found) > 1:
        found.remove("NORM")
    return found
 
 
def active_labels(labels: list[str], trained: list[str] | None, metrics: dict | None) -> list[str]:
    """Labels the model was actually trained/evaluated on. A label no source annotated (e.g. SHD) never
    receives a training signal, so its output is meaningless and must not be reported."""
    if trained:
        return [l for l in labels if l in set(trained)]
    seen: set[str] = set()
    for split in (metrics or {}).values():
        if isinstance(split, dict):
            seen |= set(split.get("per_label_auroc", {}))
    return [l for l in labels if l in seen] or list(labels)