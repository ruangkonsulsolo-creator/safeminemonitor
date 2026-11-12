"""
Download front-end and model assets once (so the demo works offline).

    python scripts/download_assets.py

* Chart.js  -> backend/static/vendor/chart.umd.min.js
* yolo11n.pt (COCO fallback model) -> models/yolo11n.pt   (via ultralytics)
If a download is blocked on your network, download the files manually in a
browser and put them in the same paths.
"""
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHART_URLS = [
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    "https://unpkg.com/chart.js@4.4.1/dist/chart.umd.min.js",
    "https://cdn.bootcdn.net/ajax/libs/Chart.js/4.4.1/chart.umd.min.js",   # mirror reachable in China
]


def chartjs():
    dest = ROOT / "backend" / "static" / "vendor" / "chart.umd.min.js"
    if dest.exists():
        print("Chart.js already present")
        return
    for u in CHART_URLS:
        try:
            with urllib.request.urlopen(u, timeout=20) as r, dest.open("wb") as f:
                shutil.copyfileobj(r, f)
            print("Chart.js downloaded from", u)
            return
        except Exception as e:
            print("failed:", u, e)
    print("!! Download Chart.js manually to", dest)


def yolo():
    try:
        from ultralytics import YOLO
        m = YOLO("yolo11n.pt")        # downloads into the current dir / ultralytics cache
        print("fallback model ready:", m.ckpt_path if hasattr(m, "ckpt_path") else "yolo11n.pt")
    except Exception as e:
        print("!! could not fetch yolo11n.pt:", e)


if __name__ == "__main__":
    chartjs()
    yolo()
