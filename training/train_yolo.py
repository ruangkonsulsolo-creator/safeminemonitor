"""
Train the PPE / unsafe-behaviour detector (YOLO11) - Phase 2 model development.

Recommended dataset: Roboflow Universe "Construction Site Safety"
(roboflow-universe-projects/construction-site-safety, CC BY 4.0), classes:
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle.  The same dataset is used by
Alotaibi & Ma (2025, Buildings 15, 4205) - one of the thesis references.

Option A (Roboflow API, needs a free API key):
    python training/train_yolo.py --roboflow-key YOUR_KEY
Option B (already-downloaded YOLO-format dataset with a data.yaml):
    python training/train_yolo.py --data datasets/css/data.yaml

GPU strongly recommended (Google Colab / Kaggle / AutoDL).  On CPU use
--model yolo11n.pt --epochs 20 --imgsz 512 (several hours).
After training, best.pt is copied to models/best.pt automatically.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_roboflow(key, version):
    from roboflow import Roboflow   # pip install roboflow
    rf = Roboflow(api_key=key)
    proj = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    ds = proj.version(version).download("yolov11", location=str(ROOT / "datasets" / "css"))
    return str(Path(ds.location) / "data.yaml")


def main(a):
    from ultralytics import YOLO
    data = a.data or get_roboflow(a.roboflow_key, a.version)
    model = YOLO(a.model)
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=15,
                project=str(ROOT / "training" / "runs"), name=a.name, exist_ok=True,
                seed=0, deterministic=True, plots=True)
    best = ROOT / "training" / "runs" / a.name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "best.pt")
    print("copied", best, "-> models/best.pt")
    # evaluation on the test split (if present)
    try:
        r = YOLO(str(best)).val(data=data, split="test", plots=True,
                                project=str(ROOT / "training" / "runs"), name=a.name + "_test", exist_ok=True)
        print(f"TEST  P={r.box.mp:.3f}  R={r.box.mr:.3f}  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")
    except Exception as e:
        print("test split evaluation skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to data.yaml (YOLO format)")
    ap.add_argument("--roboflow-key")
    ap.add_argument("--version", type=int, default=30)
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt (fast) / yolo11s.pt (better)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="ppe_yolo11")
    main(ap.parse_args())
