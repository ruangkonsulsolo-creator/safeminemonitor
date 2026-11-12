"""
SafeMine Monitor - FastAPI application.

Run (from the project root):
    uvicorn backend.app.main:app --reload --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import database as db
from .config import (UPLOAD_DIR, OUTPUT_DIR, SAMPLE_DIR, STATIC_DIR, GROUND_TRUTH_DIR, RESEARCH_DIR,
                     CATEGORIES, ROOT_DIR)
from .services import analytics, egt, risk, validation
from .services.detector import detector, capabilities_from_classes
from .services.pipeline import LIVE, submit, start_worker

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@asynccontextmanager
async def lifespan(_app):
    db.init_db()
    start_worker()
    yield


app = FastAPI(title="SafeMine Monitor API", version="0.1.0", lifespan=lifespan,
              description="Python-based computational and vision-enabled web monitoring system "
                          "for unsafe behaviour analysis in mining construction.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=OUTPUT_DIR), name="media")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ------------------------------------------------------------------ model
@app.get("/api/model")
def model_info():
    try:
        detector.load()
    except Exception as e:
        return {"loaded": False, "error": str(e)}
    i = detector.info
    return {"loaded": True, "name": i.name, "is_custom": i.is_custom, "classes": i.classes,
            "capabilities": i.capabilities, "categories": CATEGORIES}


# ------------------------------------------------------------------ upload / jobs
def _media_type(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    raise HTTPException(400, f"Unsupported file type {ext}")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), camera: str = Form("CAM-01 Demo Area"),
                 zone: str = Form("[]")):
    mtype = _media_type(file.filename)
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    dest = UPLOAD_DIR / f"{job_id}{Path(file.filename).suffix.lower()}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    zone_pts = _parse_zone(zone)
    db.create_job(job_id, file.filename, mtype, camera, zone_pts)
    pos = submit(job_id, dest, mtype, camera, zone_pts)
    return {"job_id": job_id, "media_type": mtype, "queue_position": pos}


@app.get("/api/samples")
def samples():
    files = sorted(p.name for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
    return {"samples": files}


@app.post("/api/samples/process")
def process_sample(name: str = Form(...), camera: str = Form("CAM-01 Demo Area"), zone: str = Form("[]")):
    src = (SAMPLE_DIR / name).resolve()
    if src.parent != SAMPLE_DIR.resolve() or not src.exists():
        raise HTTPException(404, "Sample not found")
    mtype = _media_type(name)
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    zone_pts = _parse_zone(zone)
    db.create_job(job_id, name, mtype, camera, zone_pts)
    pos = submit(job_id, src, mtype, camera, zone_pts)
    return {"job_id": job_id, "media_type": mtype, "queue_position": pos}


@app.get("/api/samples/{name}/frame")
def sample_first_frame(name: str):
    """First frame of a sample (used to draw the restricted zone before processing)."""
    src = (SAMPLE_DIR / name).resolve()
    if src.parent != SAMPLE_DIR.resolve() or not src.exists():
        raise HTTPException(404)
    if src.suffix.lower() in IMAGE_EXT:
        return FileResponse(src)
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "Cannot read video")
    _, jpg = cv2.imencode(".jpg", frame)
    return StreamingResponse(iter([jpg.tobytes()]), media_type="image/jpeg")


def _parse_zone(zone: str):
    try:
        pts = json.loads(zone or "[]")
        return [[float(x), float(y)] for x, y in pts][:20]
    except (ValueError, TypeError):
        return []


@app.get("/api/jobs")
def jobs():
    return db.list_jobs()


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    live = LIVE.get(job_id, {})
    j["live"] = {"t": live.get("t"), "events": live.get("events"), "workers": live.get("workers", [])}
    return j


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    db.delete_job(job_id)
    shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    ev = db.list_events(job_id)
    for e in ev:
        e["label"] = CATEGORIES.get(e["category"], e["category"])
        e["recommendation"] = risk.recommendation(e["category"], e["risk_level"])
    return ev


def _mjpeg(gen):
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/jobs/{job_id}/stream")
def live_stream(job_id: str):
    """MJPEG stream of the frames being analysed right now (monitoring view)."""
    def gen():
        last = None
        idle = 0
        while True:
            live = LIVE.get(job_id)
            if live and live.get("jpeg") is not None and live["jpeg"] is not last:
                last = live["jpeg"]
                idle = 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
            else:
                idle += 1
            if (live and live.get("done")) or idle > 3000:
                break
            time.sleep(0.04)
    return _mjpeg(gen())


@app.get("/api/jobs/{job_id}/replay")
def replay(job_id: str):
    """Re-plays the annotated result video at its native rate (codec-independent)."""
    j = db.get_job(job_id)
    if not j or not j.get("output_path"):
        raise HTTPException(404)
    path = OUTPUT_DIR / job_id / Path(j["output_path"]).name

    def gen():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 12
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1.0 / fps)
        cap.release()
    return _mjpeg(gen())


# ------------------------------------------------------------------ dashboard
@app.get("/api/dashboard")
def dashboard():
    jobs_ = db.list_jobs(500)
    events = db.list_events(limit=5000)
    for e in events:
        e["label"] = CATEGORIES.get(e["category"], e["category"])
    by_cat = {c: 0 for c in CATEGORIES}
    by_level = {"Low": 0, "Medium": 0, "High": 0}
    for e in events:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        by_level[e["risk_level"]] = by_level.get(e["risk_level"], 0) + 1
    ind = analytics.detection_indicators([j for j in jobs_ if j["status"] == "done"], events)
    return {
        "totals": {"jobs": len(jobs_), "jobs_done": sum(j["status"] == "done" for j in jobs_),
                   "events": len(events), "high_risk": by_level["High"],
                   "workers_observed": sum((j.get("summary") or {}).get("unique_workers", 0) for j in jobs_),
                   "unsafe_behaviour_rate": ind["unsafe_behaviour_rate"],
                   "helmet_compliance_rate": ind["helmet_compliance_rate"]},
        "by_category": by_cat, "by_level": by_level,
        "categories": CATEGORIES,
        "recent_events": events[:12],
        "history": [{"id": j["id"], "created_at": j["created_at"], "camera": j["camera"],
                     "filename": j["filename"], "status": j["status"],
                     "events": (j.get("summary") or {}).get("events_total"),
                     "overall_risk": ((j.get("summary") or {}).get("risk") or {}).get("overall_level")}
                    for j in jobs_[:20]],
    }


# ------------------------------------------------------------------ risk
@app.get("/api/risk")
def risk_overview():
    events = db.list_events(limit=5000)
    return {"matrix": risk.matrix_definition(), "summary": risk.site_risk_summary(events),
            "cells": _matrix_cells(events)}


def _matrix_cells(events):
    cells = {}
    for e in events:
        k = f"{e['likelihood']}-{e['severity']}"
        cells[k] = cells.get(k, 0) + 1
    return cells


# ------------------------------------------------------------------ analytics (Phase 1)
@app.get("/api/analytics")
def analytics_report():
    jobs_ = [j for j in db.list_jobs(500) if j["status"] == "done"]
    return analytics.full_report(jobs_, db.list_events(limit=10000))


@app.post("/api/analytics/upload")
async def analytics_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "CSV required")
    dest = RESEARCH_DIR / "incidents.csv"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved": dest.name}


# ------------------------------------------------------------------ EGT
@app.get("/api/egt/defaults")
def egt_defaults():
    from dataclasses import asdict
    jobs_ = [j for j in db.list_jobs(500) if j["status"] == "done"]
    ind = analytics.detection_indicators(jobs_, [])
    val = _load_validation()
    recall = (val or {}).get("overall", {}).get("recall")
    return {"params": asdict(egt.EGTParams()), "param_doc": egt.PARAM_DOC, "scenarios": egt.SCENARIOS,
            "calibration": egt.calibrate_from_system(ind["unsafe_behaviour_rate"], recall),
            "calibration_source": {"unsafe_behaviour_rate": ind["unsafe_behaviour_rate"],
                                   "recall": recall}}


@app.post("/api/egt/simulate")
def egt_simulate(body: dict):
    params = body.get("params", {})
    init = body.get("initial", {})
    sens = body.get("sensitivity", {"param": "Fw", "range": [0, 12, 13]})
    return egt.run(params, float(init.get("x0", .5)), float(init.get("y0", .5)), float(init.get("z0", .5)),
                   float(body.get("t_max", 50)), sens.get("param"), tuple(sens.get("range", [0, 12, 13])))


# ------------------------------------------------------------------ validation
VALIDATION_FILE = OUTPUT_DIR / "validation_latest.json"


def _load_validation():
    if VALIDATION_FILE.exists():
        return json.loads(VALIDATION_FILE.read_text())
    return None


@app.post("/api/validation/{job_id}")
async def run_validation(job_id: str, file: UploadFile = File(...)):
    gt_path = GROUND_TRUTH_DIR / f"{job_id}.csv"
    with gt_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    truth = validation.load_ground_truth(gt_path)
    detected = db.list_events(job_id)
    res = validation.evaluate_events(detected, truth)
    res["job_id"] = job_id
    VALIDATION_FILE.write_text(json.dumps(res, indent=2))
    return res


@app.get("/api/validation")
def get_validation():
    model_metrics = OUTPUT_DIR / "model_metrics.json"
    return {"event_level": _load_validation(),
            "model_level": json.loads(model_metrics.read_text()) if model_metrics.exists() else None}


@app.get("/api/health")
def health():
    return {"status": "ok", "root": str(ROOT_DIR)}


@app.exception_handler(RuntimeError)
async def runtime_error(_, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


_ = capabilities_from_classes  # re-export for scripts
