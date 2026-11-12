"""
Model-level validation on the held-out TEST split (Ultralytics val).

Usage:
    python scripts/evaluate_model.py --data datasets/css/data.yaml --weights models/best.pt
Writes data/outputs/model_metrics.json (shown on the Validation page).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(data, weights, split):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=data, split=split, imgsz=640, conf=0.001, iou=0.6, plots=True,
              project=str(ROOT / "data" / "outputs"), name="model_val", exist_ok=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        per_class[names[int(c)]] = round(float(r.box.ap50[i]), 3)
    out = {
        "overall": {"weights": Path(weights).name, "split": split,
                    "precision": round(float(r.box.mp), 3), "recall": round(float(r.box.mr), 3),
                    "mAP50": round(float(r.box.map50), 3), "mAP50_95": round(float(r.box.map), 3)},
        "per_class": per_class,
    }
    p = ROOT / "data" / "outputs" / "model_metrics.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved ->", p, "| plots (PR curve, confusion matrix) ->", ROOT / "data/outputs/model_val")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    main(a.data, a.weights, a.split)
