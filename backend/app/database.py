"""SQLite persistence for jobs (uploaded media) and unsafe-behaviour events."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    filename TEXT,
    media_type TEXT,            -- video | image
    camera TEXT,                -- virtual camera / location label
    status TEXT,                -- queued | processing | done | error
    progress REAL DEFAULT 0,
    model_name TEXT,
    fps REAL,
    duration_s REAL,
    frames_total INTEGER,
    frames_processed INTEGER,
    person_frames INTEGER DEFAULT 0,        -- sum over frames of persons seen
    unsafe_person_frames INTEGER DEFAULT 0, -- persons with >=1 violation
    zone TEXT,                  -- JSON polygon (normalised)
    output_path TEXT,
    summary TEXT,               -- JSON
    error TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    camera TEXT,
    category TEXT,
    track_id INTEGER,
    start_s REAL,
    end_s REAL,
    duration_s REAL,
    max_conf REAL,
    exposure_ratio REAL,
    likelihood INTEGER,
    severity INTEGER,
    risk_score INTEGER,
    risk_level TEXT,
    snapshot TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _lock, get_conn() as c:
        c.executescript(SCHEMA)


def now():
    return datetime.now().isoformat(timespec="seconds")


def create_job(job_id, filename, media_type, camera, zone):
    with _lock, get_conn() as c:
        c.execute(
            "INSERT INTO jobs (id, created_at, filename, media_type, camera, status, zone) "
            "VALUES (?,?,?,?,?,?,?)",
            (job_id, now(), filename, media_type, camera, "queued", json.dumps(zone or [])),
        )


def update_job(job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    with _lock, get_conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*vals, job_id))


def get_job(job_id):
    with get_conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(r)


def list_jobs(limit=100):
    with get_conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def insert_events(events):
    if not events:
        return
    keys = ["job_id", "camera", "category", "track_id", "start_s", "end_s", "duration_s",
            "max_conf", "exposure_ratio", "likelihood", "severity", "risk_score",
            "risk_level", "snapshot"]
    with _lock, get_conn() as c:
        c.executemany(
            f"INSERT INTO events ({', '.join(keys)}, created_at) VALUES ({', '.join('?' * len(keys))}, ?)",
            [tuple(e.get(k) for k in keys) + (now(),) for e in events],
        )


def list_events(job_id=None, limit=1000):
    q = "SELECT * FROM events"
    args = []
    if job_id:
        q += " WHERE job_id=?"
        args.append(job_id)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id):
    with _lock, get_conn() as c:
        c.execute("DELETE FROM events WHERE job_id=?", (job_id,))
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for k in ("summary", "zone"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                pass
    return d
