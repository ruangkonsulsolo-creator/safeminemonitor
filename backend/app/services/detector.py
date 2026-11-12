"""
Computer-vision detector (Phase 2).

Wraps an Ultralytics YOLO model with ByteTrack multi-object tracking so that
each worker keeps a stable ID across video frames.  The detector returns a
plain list of dicts so the rule engine and the tests do not depend on
ultralytics.

Model selection:
    1. models/best.pt  -> custom PPE model (recommended, trained in training/)
    2. yolo11n.pt      -> generic COCO model (person + vehicles only)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import (CUSTOM_MODEL_PATH, FALLBACK_MODEL, CONF_THRESHOLD, IOU_THRESHOLD,
                      IMG_SIZE, TRACKER_CFG, PERSON_CLASSES, NO_HELMET_CLASSES,
                      HELMET_CLASSES, NO_VEST_CLASSES, VEST_CLASSES, MACHINE_CLASSES)

log = logging.getLogger("safemine.detector")


@dataclass
class ModelInfo:
    name: str
    path: str
    classes: dict
    is_custom: bool
    capabilities: dict = field(default_factory=dict)


def role_of(class_name: str) -> str | None:
    n = class_name.lower().strip()
    if n in PERSON_CLASSES:
        return "person"
    if n in NO_HELMET_CLASSES:
        return "no_helmet"
    if n in HELMET_CLASSES:
        return "helmet"
    if n in NO_VEST_CLASSES:
        return "no_vest"
    if n in VEST_CLASSES:
        return "vest"
    if n in MACHINE_CLASSES:
        return "machine"
    return None


def capabilities_from_classes(names: dict) -> dict:
    roles = {role_of(v) for v in names.values()}
    return {
        "NO_HELMET": "no_helmet" in roles or "helmet" in roles,
        "NO_VEST": "no_vest" in roles or "vest" in roles,
        "RESTRICTED_ZONE": "person" in roles,
        "MACHINE_PROXIMITY": "person" in roles and "machine" in roles,
    }


class Detector:
    """Lazy-loaded YOLO + ByteTrack detector."""

    def __init__(self):
        self._model = None
        self.info: ModelInfo | None = None

    def load(self):
        if self._model is not None:
            return self
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed: pip install -r requirements.txt") from e

        if CUSTOM_MODEL_PATH.exists():
            path, custom = str(CUSTOM_MODEL_PATH), True
        else:
            log.warning("Custom model %s not found - falling back to %s (COCO). "
                        "PPE categories will be disabled.", CUSTOM_MODEL_PATH, FALLBACK_MODEL)
            path, custom = FALLBACK_MODEL, False
        self._model = YOLO(path)
        names = dict(self._model.names)
        self.info = ModelInfo(name=path.split("/")[-1], path=path, classes=names,
                              is_custom=custom, capabilities=capabilities_from_classes(names))
        log.info("Loaded model %s with classes %s", path, names)
        return self

    def reset_tracker(self):
        """Call before each new video so track IDs restart."""
        if self._model is not None and getattr(self._model, "predictor", None) is not None:
            try:
                for t in getattr(self._model.predictor, "trackers", []) or []:
                    t.reset()
            except Exception:  # pragma: no cover - tracker API differs between versions
                self._model.predictor = None

    def detect(self, frame, track: bool = True) -> list[dict]:
        """Run detection (+ tracking) on one BGR frame."""
        self.load()
        if track:
            res = self._model.track(frame, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESHOLD,
                                    iou=IOU_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
        else:
            res = self._model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                      imgsz=IMG_SIZE, verbose=False)[0]
        out = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, p, tid in zip(xyxy, cls, conf, ids):
            name = self.info.classes.get(int(c), str(c))
            out.append({
                "bbox": [float(v) for v in b],
                "class_id": int(c),
                "class_name": name,
                "role": role_of(name),
                "conf": float(p),
                "track_id": int(tid) if tid is not None else None,
            })
        return out


detector = Detector()
