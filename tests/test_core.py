"""
Unit / integration tests that run WITHOUT ultralytics (mock detector).
    python -m pytest -q        or        python tests/test_core.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services import egt, risk, validation  # noqa: E402
from backend.app.services.rules import evaluate_frame, EventAggregator, point_in_polygon  # noqa: E402

CAPS = {"NO_HELMET": True, "NO_VEST": True, "RESTRICTED_ZONE": True, "MACHINE_PROXIMITY": True}


def det(role, name, box, conf=0.9, tid=None):
    return {"role": role, "class_name": name, "bbox": box, "conf": conf, "track_id": tid}


def test_ppe_assignment():
    dets = [det("person", "Person", [100, 100, 200, 400], tid=1),
            det("no_helmet", "NO-Hardhat", [130, 100, 170, 140]),
            det("vest", "Safety Vest", [110, 160, 190, 260])]
    w, _ = evaluate_frame(dets, 640, 480, None, CAPS)
    assert w[0]["helmet"] == "No" and w[0]["vest"] == "Yes"
    assert "NO_HELMET" in w[0]["violations"] and "NO_VEST" not in w[0]["violations"]


def test_zone_and_proximity():
    zone = [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]
    dets = [det("person", "person", [400, 200, 450, 400], tid=2),
            det("machine", "truck", [460, 250, 600, 420])]
    w, m = evaluate_frame(dets, 640, 480, zone, CAPS)
    assert w[0]["in_zone"] and w[0]["near_machine"]
    assert point_in_polygon((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)])


def test_operator_in_cab_not_flagged():
    dets = [det("person", "person", [480, 260, 520, 330], tid=3), det("machine", "truck", [460, 250, 600, 420])]
    w, _ = evaluate_frame(dets, 640, 480, None, CAPS)
    assert not w[0]["near_machine"]


def test_event_aggregation_and_risk():
    agg = EventAggregator(fps=10)
    for k in range(40):                       # 4 s of violation then 2 s compliant
        t = k / 10
        viol = {"NO_HELMET": 0.8} if t < 4 else {}
        agg.update(t, [{"track_id": 1, "violations": viol}])
    for k in range(40, 60):
        agg.update(k / 10, [{"track_id": 1, "violations": {}}])
    ev = agg.finish()
    assert len(ev) == 1 and abs(ev[0]["duration_s"] - 3.9) < 0.11
    risk.assess_events(ev)
    assert ev[0]["severity"] == 3 and ev[0]["likelihood"] >= 4 and ev[0]["risk_level"] in ("Medium", "High")


def test_short_flicker_ignored():
    agg = EventAggregator(fps=10)
    agg.update(0.0, [{"track_id": 1, "violations": {"NO_VEST": 0.5}}])
    for k in range(1, 30):
        agg.update(k / 10, [{"track_id": 1, "violations": {}}])
    assert agg.finish() == []


def test_validation_metrics():
    d = [{"category": "NO_HELMET", "start_s": 1, "end_s": 5}, {"category": "NO_HELMET", "start_s": 20, "end_s": 21}]
    g = [{"category": "NO_HELMET", "start_s": 1.5, "end_s": 6}, {"category": "RESTRICTED_ZONE", "start_s": 8, "end_s": 9}]
    r = validation.evaluate_events(d, g)
    assert r["overall"]["TP"] == 1 and r["overall"]["FP"] == 1 and r["overall"]["FN"] == 1


def test_egt_math():
    p = egt.EGTParams()
    # replicator vanishes at every vertex
    for v in [(0, 0, 0), (1, 1, 1), (1, 0, 1)]:
        assert np.allclose(egt.replicator(p, np.array(v, float)), 0)
    # finite-difference check of the vertex eigenvalue formula
    v = np.array([0.0, 1.0, 1.0])
    F = egt.fitness_gaps(p, *v)
    h = 1e-6
    num = (egt.replicator(p, v + [h, 0, 0])[0] - egt.replicator(p, v)[0]) / h
    assert abs(num - F[0]) < 1e-4
    out = egt.run({}, sens_param=None)
    assert 0 <= out["final"]["x"] <= 1 and len(out["equilibria"]) == 8


def test_pipeline_with_mock_detector(tmp_path=None):
    """End-to-end: synthetic video -> pipeline -> DB events (mock detector)."""
    import tempfile
    from backend.app import database as db
    from backend.app.services import pipeline
    from backend.app.services.detector import ModelInfo

    class Mock:
        info = ModelInfo("mock", "mock", {0: "Person", 1: "NO-Hardhat"}, True, CAPS)
        def load(self): return self
        def reset_tracker(self): pass
        def detect(self, frame, track=True):
            i = int(frame[0, 0, 0])            # frame index encoded in pixel
            x = 50 + 3 * i
            out = [det("person", "Person", [x, 100, x + 60, 300], 0.9, 1)]
            if 10 <= i < 40:
                out.append(det("no_helmet", "NO-Hardhat", [x + 15, 100, x + 45, 130], 0.85))
            return out

    pipeline.detector = Mock()
    d = Path(tempfile.mkdtemp())
    vid = d / "synthetic.avi"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 320))
    for i in range(60):
        f = np.zeros((320, 320, 3), np.uint8)
        f[0, 0] = i
        wr.write(f)
    wr.release()
    db.init_db()
    jid = "test-" + str(np.random.randint(1e9))
    db.create_job(jid, "synthetic.avi", "video", "TEST", [])
    old = pipeline.FRAME_STRIDE
    pipeline.FRAME_STRIDE = 1
    pipeline.process_video(jid, vid, "TEST", [])
    pipeline.FRAME_STRIDE = old
    j = db.get_job(jid)
    ev = db.list_events(jid)
    assert j["status"] == "done" and len(ev) == 1 and ev[0]["category"] == "NO_HELMET"
    print("pipeline summary:", j["summary"]["events_total"], "events,", j["summary"]["risk"]["overall_level"])
    db.delete_job(jid)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
