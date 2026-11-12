# SafeMine Monitor

**A Python-Based Computational Framework and Web-Based Monitoring System for Unsafe Behavior Analysis in Mining Construction**
(preliminary research prototype – Master's thesis, Safety Science and Engineering)

Flow implemented:
`Upload video/image → Python processing → YOLO11 detection + tracking → unsafe-behaviour rules → time-stamped events → HIRARC risk → database → web monitoring`
plus Phase-1 Python analytics and a tripartite Evolutionary Game Theory simulation calibrated by the vision results.

## Quick start (Windows / macOS / Linux, Python 3.10–3.12)
```bash
cd SafeMine_Monitor
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_assets.py            # Chart.js + yolo11n.pt (once)
python scripts/generate_synthetic_incidents.py   # demo research data (replace later)
uvicorn backend.app.main:app --port 8000
```
Open http://127.0.0.1:8000 → *Upload & Detect* → choose a video → (optional) draw a zone → **Run detection**.
In China, if pip is slow: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`.

## Folder structure
```
backend/app/main.py            FastAPI routes (upload, jobs, stream, dashboard, risk, analytics, EGT, validation)
backend/app/config.py          all thresholds & paths (report these in the thesis)
backend/app/database.py        SQLite (jobs, events)
backend/app/services/
    detector.py                YOLO11 + ByteTrack wrapper, model capability detection
    rules.py                   unsafe-behaviour rules + temporal event aggregation
    pipeline.py                video/image processing, annotated output, live MJPEG buffer
    risk.py                    HIRARC L×S matrix, recommendations (ISO 45001 hierarchy)
    analytics.py               Phase-1 statistics & behavioural risk indicators
    egt.py                     tripartite EGT replicator dynamics, stability, sensitivity
    validation.py              event-level precision / recall / F1
backend/static/                web UI (index.html, css, js)
training/                      YOLO training script + Colab notebook
scripts/                       batch processing, MSHA preparation, model evaluation, figures
data/samples                   demo videos ("virtual cameras")
data/research                  research CSV (Phase 1)
data/ground_truth              manual labels for validation
docs/                          architecture figures, methodology notes
tests/test_core.py             tests (run without GPU/ultralytics: python tests/test_core.py)
```

## No live camera? (upload-only limitation)
The system treats every uploaded video as a **virtual camera**. While it is analysed, the annotated
frames are streamed to the browser (MJPEG) with worker IDs and a per-worker safety panel, i.e. the
same view an operator would see on a live feed, clearly labelled **REPLAY**. Results are stored with
video-time stamps. The video source is a single `cv2.VideoCapture` call, so a webcam (`0`) or an RTSP
CCTV URL can be plugged in later without changing the rest of the pipeline (future work).

## Honest scope statement
* Working: upload → detection → rules → events → risk → dashboard; analytics; EGT; validation.
* PPE categories need the custom model `models/best.pt`; without it only zone/proximity rules run.
* Unsafe operation / movement categories (action recognition) are future work.
* EGT parameters are illustrative until justified by literature or expert survey.
