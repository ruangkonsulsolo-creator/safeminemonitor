"""
Preliminary validation of the identification results.

Two complementary levels are used in the thesis:

1. Object-detection level (model quality) - computed with Ultralytics
   `model.val()` on the held-out test split of the dataset
   (scripts/evaluate_model.py): Precision, Recall, mAP@0.5, mAP@0.5:0.95.

2. Event level (system quality, this module) - detected unsafe-behaviour
   events of a test video are compared with manually annotated ground-truth
   intervals (data/ground_truth/<video>.csv):

        category,start_s,end_s,note
        NO_HELMET,3.0,9.5,worker on the left
        RESTRICTED_ZONE,12.0,15.0,

   A detection is a True Positive if it has the same category and a temporal
   IoU >= TIOU_THRESHOLD with a still-unmatched ground-truth interval
   (greedy one-to-one matching, highest IoU first).
"""
from __future__ import annotations

import csv
from pathlib import Path

TIOU_THRESHOLD = 0.3


def t_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:          # both are instants (images) -> match if identical time
        return 1.0 if abs(a[0] - b[0]) < 1e-6 else 0.0
    return inter / union


def load_ground_truth(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("category"):
                continue
            rows.append({"category": r["category"].strip().upper(),
                         "start_s": float(r.get("start_s") or 0), "end_s": float(r.get("end_s") or 0)})
    return rows


def evaluate_events(detected: list[dict], truth: list[dict], thr: float = TIOU_THRESHOLD) -> dict:
    cats = sorted({e["category"] for e in detected} | {g["category"] for g in truth})
    per_cat, TP, FP, FN = {}, 0, 0, 0
    matches = []
    for c in cats:
        D = [e for e in detected if e["category"] == c]
        G = [g for g in truth if g["category"] == c]
        pairs = sorted(((t_iou((d["start_s"], d["end_s"]), (g["start_s"], g["end_s"])), i, j)
                        for i, d in enumerate(D) for j, g in enumerate(G)), reverse=True)
        used_d, used_g = set(), set()
        for iou_v, i, j in pairs:
            if iou_v < thr or i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            matches.append({"category": c, "detected": [D[i]["start_s"], D[i]["end_s"]],
                            "truth": [G[j]["start_s"], G[j]["end_s"]], "t_iou": round(iou_v, 3)})
        tp, fp, fn = len(used_d), len(D) - len(used_d), len(G) - len(used_g)
        per_cat[c] = _prf(tp, fp, fn)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    return {"threshold_tiou": thr, "overall": _prf(TP, FP, FN), "per_category": per_cat, "matches": matches}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
