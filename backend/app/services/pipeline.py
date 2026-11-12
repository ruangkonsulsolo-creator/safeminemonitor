"""
Processing pipeline:  Upload -> Python/CV processing -> Detection -> Results -> Web.

A single background worker thread consumes the job queue (YOLO tracking state
is not thread-safe, and a sequential queue is enough for a research prototype).
While a video is processed, the latest annotated frame and worker states are
kept in LIVE[job_id] so the browser can watch the analysis as it happens
("replay monitoring" - the substitute for a live camera feed).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .. import database as db
from ..config import OUTPUT_DIR, FRAME_STRIDE, MAX_VIDEO_SECONDS, CATEGORIES
from . import risk
from .detector import detector
from .rules import evaluate_frame, EventAggregator, image_events

log = logging.getLogger("safemine.pipeline")

LIVE: dict[str, dict] = {}
_jobs: "queue.Queue[tuple]" = queue.Queue()

COLORS = {  # BGR
    "ok": (80, 200, 80), "NO_HELMET": (40, 40, 230), "NO_VEST": (0, 140, 255),
    "RESTRICTED_ZONE": (200, 60, 200), "MACHINE_PROXIMITY": (0, 0, 180), "machine": (0, 215, 255),
}
SHORT = {"NO_HELMET": "NO HELMET", "NO_VEST": "NO VEST", "RESTRICTED_ZONE": "RESTRICTED ZONE",
         "MACHINE_PROXIMITY": "MACHINE PROXIMITY"}


# ------------------------------------------------------------------ drawing
def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw(frame, workers, machines, zone_norm, camera, t, n_events):
    img = frame.copy()
    h, w = img.shape[:2]
    if zone_norm and len(zone_norm) >= 3:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone_norm], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (60, 60, 220))
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.polylines(img, [pts], True, (60, 60, 220), 2)
        _label(img, "RESTRICTED ZONE", int(pts[:, 0].min()), int(pts[:, 1].min()), (60, 60, 220))
    for m in machines:
        x1, y1, x2, y2 = map(int, m["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["machine"], 2)
        _label(img, f"{m['class_name']} {m['conf']:.2f}", x1, y1, (0, 150, 190))
    for wk in workers:
        x1, y1, x2, y2 = map(int, wk["bbox"])
        v = list(wk["violations"])
        color = COLORS[v[0]] if v else COLORS["ok"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _label(img, f"Worker {wk['track_id']}", x1, y1, color)
        for i, c in enumerate(v):
            _label(img, SHORT[c], x1, y2 + 18 * (i + 1), COLORS[c])
    # HUD
    cv2.rectangle(img, (0, 0), (w, 28), (20, 20, 20), -1)
    mm, ss = divmod(t, 60)
    cv2.putText(img, f"SafeMine Monitor | {camera} | REPLAY t={int(mm):02d}:{ss:05.2f}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    txt = f"workers {len(workers)} | events {n_events}"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, txt, (w - tw - 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return img


def _open_writer(base: Path, fps: float, size):
    """Browser-friendly codec first (H.264), then VP8/WebM, then MPEG-4."""
    for fourcc, ext in (("avc1", ".mp4"), ("VP80", ".webm"), ("mp4v", ".mp4")):
        path = base.with_suffix(ext)
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if wr.isOpened():
            return wr, path
        wr.release()
    raise RuntimeError("No usable video codec found in OpenCV")


# ------------------------------------------------------------------ processing
def _finish(job_id, camera, events, extra_summary, workers_seen, model_info):
    risk.assess_events(events)
    for e in events:
        e["job_id"], e["camera"] = job_id, camera
    db.insert_events(events)
    counts = Counter(e["category"] for e in events)
    summary = {
        **extra_summary,
        "events_total": len(events),
        "events_by_category": {c: counts.get(c, 0) for c in CATEGORIES},
        "unique_workers": len(workers_seen),
        "risk": risk.site_risk_summary(events),
        "model": model_info.name if model_info else None,
        "model_is_custom": model_info.is_custom if model_info else None,
        "capabilities": model_info.capabilities if model_info else {},
    }
    return summary


def process_video(job_id, path: Path, camera: str, zone):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(total or 10 ** 9, fps * MAX_VIDEO_SECONDS))
    detector.load()
    detector.reset_tracker()
    caps = detector.info.capabilities
    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    writer, out_path = None, None
    agg = EventAggregator(fps)
    stats = Counter()
    workers_seen = set()
    db.update_job(job_id, status="processing", fps=fps, frames_total=max_frames,
                  duration_s=round(max_frames / fps, 2), model_name=detector.info.name)
    LIVE[job_id] = {"jpeg": None, "workers": [], "t": 0.0, "events": 0, "done": False}

    idx = processed = 0
    current = {"img": None}

    def snapshot(tid, cat, t):
        name = f"snap_{tid}_{cat}_{t:.1f}.jpg"
        cv2.imwrite(str(out_dir / name), current["img"])
        return f"/media/{job_id}/{name}"

    t0 = time.time()
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STRIDE:
            idx += 1
            continue
        t = idx / fps
        h, w = frame.shape[:2]
        dets = detector.detect(frame, track=True)
        workers, machines = evaluate_frame(dets, w, h, zone, caps)
        n_ev = len(agg.closed) + len(agg.open)
        img = draw(frame, workers, machines, zone, camera, t, n_ev)
        current["img"] = img
        agg.update(t, workers, snapshot)

        for wk in workers:
            workers_seen.add(wk["track_id"])
            stats["person_frames"] += 1
            if wk["violations"]:
                stats["unsafe_person_frames"] += 1
            if "NO_HELMET" in wk["violations"]:
                stats["helmet_violation_person_frames"] += 1
            if "NO_VEST" in wk["violations"]:
                stats["vest_violation_person_frames"] += 1

        if writer is None:
            writer, out_path = _open_writer(out_dir / "annotated", fps / FRAME_STRIDE, (w, h))
        writer.write(img)
        ok_j, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        LIVE[job_id].update(jpeg=jpg.tobytes() if ok_j else None, t=round(t, 2), events=n_ev,
                            workers=[{k: wk[k] for k in ("track_id", "helmet", "vest", "in_zone",
                                                         "near_machine", "machine_dist_ratio")} |
                                     {"violations": list(wk["violations"])} for wk in workers])
        processed += 1
        idx += 1
        if processed % 10 == 0:
            db.update_job(job_id, progress=round(idx / max_frames * 100, 1), frames_processed=idx,
                          person_frames=stats["person_frames"],
                          unsafe_person_frames=stats["unsafe_person_frames"])
    cap.release()
    if writer:
        writer.release()
    events = agg.finish()
    summary = _finish(job_id, camera, events,
                      {"processing_seconds": round(time.time() - t0, 1),
                       "processing_fps": round(processed / max(time.time() - t0, 1e-6), 2),
                       "frame_stride": FRAME_STRIDE,
                       "helmet_violation_person_frames": stats["helmet_violation_person_frames"],
                       "vest_violation_person_frames": stats["vest_violation_person_frames"],
                       "unsafe_behaviour_rate": round(stats["unsafe_person_frames"] /
                                                      stats["person_frames"], 4) if stats["person_frames"] else None},
                      workers_seen, detector.info)
    db.update_job(job_id, status="done", progress=100, frames_processed=idx,
                  person_frames=stats["person_frames"], unsafe_person_frames=stats["unsafe_person_frames"],
                  output_path=f"/media/{job_id}/{out_path.name}" if out_path else None, summary=summary)
    LIVE[job_id]["done"] = True


def process_image(job_id, path: Path, camera: str, zone):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError("Cannot read image")
    detector.load()
    caps = detector.info.capabilities
    h, w = frame.shape[:2]
    db.update_job(job_id, status="processing", model_name=detector.info.name, frames_total=1)
    dets = detector.detect(frame, track=False)
    workers, machines = evaluate_frame(dets, w, h, zone, caps)
    events = image_events(workers)
    img = draw(frame, workers, machines, zone, camera, 0.0, len(events))
    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "annotated.jpg"), img)
    for e in events:
        e["snapshot"] = f"/media/{job_id}/annotated.jpg"
    pf = len(workers)
    upf = sum(1 for wk in workers if wk["violations"])
    summary = _finish(job_id, camera, events, {
        "helmet_violation_person_frames": sum(1 for wk in workers if "NO_HELMET" in wk["violations"]),
        "vest_violation_person_frames": sum(1 for wk in workers if "NO_VEST" in wk["violations"]),
        "unsafe_behaviour_rate": round(upf / pf, 4) if pf else None,
        "detections": [{k: d[k] for k in ("class_name", "conf", "bbox")} for d in dets],
        "workers": [{k: wk[k] for k in ("track_id", "helmet", "vest", "in_zone", "near_machine")} |
                    {"violations": list(wk["violations"])} for wk in workers],
    }, {wk["track_id"] for wk in workers}, detector.info)
    db.update_job(job_id, status="done", progress=100, frames_processed=1, person_frames=pf,
                  unsafe_person_frames=upf, output_path=f"/media/{job_id}/annotated.jpg", summary=summary)


def _worker_loop():
    while True:
        job_id, path, media_type, camera, zone = _jobs.get()
        try:
            if media_type == "video":
                process_video(job_id, path, camera, zone)
            else:
                process_image(job_id, path, camera, zone)
        except Exception as e:  # keep the worker alive
            log.error("Job %s failed: %s", job_id, traceback.format_exc())
            db.update_job(job_id, status="error", error=str(e))
            if job_id in LIVE:
                LIVE[job_id]["done"] = True
        finally:
            _jobs.task_done()


_started = False


def start_worker():
    global _started
    if not _started:
        threading.Thread(target=_worker_loop, daemon=True, name="safemine-worker").start()
        _started = True


def submit(job_id, path: Path, media_type, camera, zone):
    start_worker()
    _jobs.put((job_id, path, media_type, camera, zone))
    return _jobs.qsize()
