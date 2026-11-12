"""
Process all media in data/samples without the web UI (useful for the thesis
experiments and for pre-loading results before a demo).

Usage:  python scripts/batch_process.py [--zone "[[0.6,0.5],[0.95,0.5],[0.95,0.95],[0.6,0.95]]"]
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app import database as db  # noqa: E402
from backend.app.config import SAMPLE_DIR  # noqa: E402
from backend.app.services.pipeline import process_video, process_image  # noqa: E402

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(zone, camera):
    db.init_db()
    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in VIDEO | IMAGE)
    if not files:
        print("No media in", SAMPLE_DIR)
        return
    for f in files:
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        mtype = "video" if f.suffix.lower() in VIDEO else "image"
        db.create_job(jid, f.name, mtype, camera, zone)
        t = time.time()
        try:
            (process_video if mtype == "video" else process_image)(jid, f, camera, zone)
            s = db.get_job(jid)["summary"]
            print(f"[ok] {f.name}: {s['events_total']} events, UBR={s.get('unsafe_behaviour_rate')}, "
                  f"risk={s['risk']['overall_level']} ({time.time() - t:.1f}s)")
        except Exception as e:
            db.update_job(jid, status="error", error=str(e))
            print(f"[error] {f.name}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="[]", help="normalised polygon JSON")
    ap.add_argument("--camera", default="CAM-01 Batch")
    a = ap.parse_args()
    main(json.loads(a.zone), a.camera)
