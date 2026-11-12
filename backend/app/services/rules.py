"""
Unsafe-behaviour rule engine (Phase 2 -> identification layer).

Object detection alone only says "there is a helmet / a person / a truck".
This module turns raw detections into *behaviour-level* judgements for each
tracked worker, following the two-stage logic used in the reference works
(detector -> reasoning on worker state; cf. Tang 2021, Bonyaniakbarabadi 2025):

    NO_HELMET          worker box contains a "NO-Hardhat" detection
    NO_VEST            worker box contains a "NO-Safety Vest" detection
    RESTRICTED_ZONE    worker foot point lies inside a user-defined polygon
    MACHINE_PROXIMITY  worker foot point is closer than PROXIMITY_RATIO x
                       (worker height) to a machinery / vehicle box

`EventAggregator` then converts frame-level violations into time-stamped
events (start, end, duration) with a minimum-persistence filter.
"""
from __future__ import annotations

import math

from ..config import PPE_ASSIGN_IOA, PROXIMITY_RATIO, MIN_EVENT_SECONDS, EVENT_GAP_SECONDS


# ---------------------------------------------------------------- geometry
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def ioa(inner, outer):
    """Intersection over the area of `inner`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = area(inner)
    return inter / a if a > 0 else 0.0


def foot_point(b):
    return ((b[0] + b[2]) / 2.0, b[3])


def point_box_distance(p, b):
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def point_in_polygon(p, poly):
    """Ray-casting test; poly = [(x, y), ...] in pixel coordinates."""
    if not poly or len(poly) < 3:
        return False
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------- per frame
def evaluate_frame(dets: list[dict], frame_w: int, frame_h: int, zone_norm=None,
                   capabilities: dict | None = None):
    """
    Returns
        workers:    list of {track_id, bbox, conf, helmet, vest, in_zone, near_machine, violations}
        machines:   list of machine detections
    """
    caps = capabilities or {}
    persons = [d for d in dets if d["role"] == "person"]
    machines = [d for d in dets if d["role"] == "machine"]
    no_helmet = [d for d in dets if d["role"] == "no_helmet"]
    helmet = [d for d in dets if d["role"] == "helmet"]
    no_vest = [d for d in dets if d["role"] == "no_vest"]
    vest = [d for d in dets if d["role"] == "vest"]

    zone_px = [(x * frame_w, y * frame_h) for x, y in (zone_norm or [])]

    # PPE-violation boxes that do not fall inside any person box still
    # represent a worker (e.g. person class missed) -> create pseudo-workers.
    def assigned(box, pool):
        return [d for d in pool if ioa(d["bbox"], box) >= PPE_ASSIGN_IOA]

    pseudo = []
    for d in no_helmet + no_vest:
        if not any(ioa(d["bbox"], p["bbox"]) >= PPE_ASSIGN_IOA for p in persons):
            pseudo.append({**d, "role": "person", "pseudo": True})
    persons_all = persons + pseudo

    workers = []
    for i, p in enumerate(persons_all):
        b = p["bbox"]
        tid = p.get("track_id")
        if tid is None:
            tid = 10_000 + i            # image mode / untracked
        h_bad = assigned(b, no_helmet) or ([p] if p.get("pseudo") and p["class_name"].lower() in
                                           {"no-hardhat", "no_hardhat", "no-helmet", "no_helmet", "head", "no hardhat"} else [])
        v_bad = assigned(b, no_vest) or ([p] if p.get("pseudo") and "vest" in p["class_name"].lower() else [])
        h_ok = assigned(b, helmet)
        v_ok = assigned(b, vest)

        helmet_state = "No" if h_bad else ("Yes" if h_ok else "Unknown")
        vest_state = "No" if v_bad else ("Yes" if v_ok else "Unknown")

        fp = foot_point(b)
        in_zone = bool(zone_px) and point_in_polygon(fp, zone_px)

        near, min_ratio = False, None
        ph = max(1.0, b[3] - b[1])
        for m in machines:
            if ioa(b, m["bbox"]) > 0.6:      # person inside the machine box -> operator in cab
                continue
            r = point_box_distance(fp, m["bbox"]) / ph
            min_ratio = r if min_ratio is None else min(min_ratio, r)
            if r < PROXIMITY_RATIO:
                near = True

        violations = {}
        if helmet_state == "No" and caps.get("NO_HELMET", True):
            violations["NO_HELMET"] = max(d["conf"] for d in h_bad)
        if vest_state == "No" and caps.get("NO_VEST", True):
            violations["NO_VEST"] = max(d["conf"] for d in v_bad)
        if in_zone:
            violations["RESTRICTED_ZONE"] = p["conf"]
        if near:
            violations["MACHINE_PROXIMITY"] = p["conf"]

        workers.append({
            "track_id": int(tid), "bbox": b, "conf": p["conf"],
            "helmet": helmet_state, "vest": vest_state,
            "in_zone": in_zone, "near_machine": near,
            "machine_dist_ratio": None if min_ratio is None else round(min_ratio, 2),
            "violations": violations,
        })
    return workers, machines


# ---------------------------------------------------------------- events
class EventAggregator:
    """Turns frame-level violations into temporal events per (worker, category)."""

    def __init__(self, fps: float, min_event_s=MIN_EVENT_SECONDS, gap_s=EVENT_GAP_SECONDS):
        self.fps = fps or 25.0
        self.min_event_s = min_event_s
        self.gap_s = gap_s
        self.open: dict[tuple, dict] = {}
        self.closed: list[dict] = []
        self.worker_seen: dict[int, list] = {}      # track_id -> [first_t, last_t, n_frames]

    def update(self, t: float, workers: list[dict], snapshot_cb=None):
        for w in workers:
            tid = w["track_id"]
            s = self.worker_seen.setdefault(tid, [t, t, 0])
            s[1] = t
            s[2] += 1
            for cat, conf in w["violations"].items():
                key = (tid, cat)
                ev = self.open.get(key)
                if ev and t - ev["last_t"] <= self.gap_s:
                    ev["last_t"] = t
                    ev["n"] += 1
                    ev["max_conf"] = max(ev["max_conf"], conf)
                else:
                    if ev:
                        self._close(key)
                    self.open[key] = {"track_id": tid, "category": cat, "start_s": t,
                                      "last_t": t, "n": 1, "max_conf": conf, "snapshot": None}
                ev = self.open[key]
                if ev["snapshot"] is None and snapshot_cb and (t - ev["start_s"]) >= self.min_event_s:
                    ev["snapshot"] = snapshot_cb(tid, cat, t)
        # close episodes that exceeded the gap
        for key in [k for k, e in self.open.items() if t - e["last_t"] > self.gap_s]:
            self._close(key)

    def _close(self, key):
        e = self.open.pop(key)
        dur = e["last_t"] - e["start_s"]
        if dur >= self.min_event_s or self.min_event_s == 0:
            first, last, _ = self.worker_seen.get(e["track_id"], [e["start_s"], e["last_t"], 0])
            observed = max(last - first, dur, 1e-6)
            self.closed.append({
                "track_id": e["track_id"], "category": e["category"],
                "start_s": round(e["start_s"], 2), "end_s": round(e["last_t"], 2),
                "duration_s": round(dur, 2), "max_conf": round(float(e["max_conf"]), 3),
                "exposure_ratio": round(min(1.0, dur / observed), 3),
                "snapshot": e["snapshot"],
            })

    def finish(self):
        for key in list(self.open):
            self._close(key)
        # recompute exposure ratio with the final observation window of each worker
        for e in self.closed:
            first, last, _ = self.worker_seen.get(e["track_id"], [e["start_s"], e["end_s"], 0])
            observed = max(last - first, e["duration_s"], 1e-6)
            e["exposure_ratio"] = round(min(1.0, e["duration_s"] / observed), 3)
        return sorted(self.closed, key=lambda e: e["start_s"])


def image_events(workers: list[dict]) -> list[dict]:
    """For a single image: one event per worker x violation (duration 0)."""
    out = []
    for w in workers:
        for cat, conf in w["violations"].items():
            out.append({"track_id": w["track_id"], "category": cat, "start_s": 0.0, "end_s": 0.0,
                        "duration_s": 0.0, "max_conf": round(float(conf), 3),
                        "exposure_ratio": 1.0, "snapshot": None})
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


__all__ = ["evaluate_frame", "EventAggregator", "image_events", "ioa", "iou",
           "point_in_polygon"]
