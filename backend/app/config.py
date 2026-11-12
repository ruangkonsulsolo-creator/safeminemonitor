"""
SafeMine Monitor - central configuration.

All thresholds that influence detection, risk scoring and the EGT model are
kept here so they can be reported transparently in the thesis (Chapter 3,
"Parameter settings").  Change values here, not inside the service modules.
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]          # SafeMine_Monitor/
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
SAMPLE_DIR = DATA_DIR / "samples"
RESEARCH_DIR = DATA_DIR / "research"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
MODEL_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "backend" / "static"
DB_PATH = DATA_DIR / "safemine.db"

for _d in (UPLOAD_DIR, OUTPUT_DIR, SAMPLE_DIR, RESEARCH_DIR, GROUND_TRUTH_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Computer-vision model
# --------------------------------------------------------------------------
# 1st choice: custom PPE model trained on the Roboflow "Construction Site
#             Safety" dataset (classes: Hardhat, NO-Hardhat, Safety Vest,
#             NO-Safety Vest, Person, machinery, vehicle, ...).
# 2nd choice: generic COCO model (person / truck / car / bus) -> only the
#             restricted-zone and machinery-proximity rules are available.
CUSTOM_MODEL_PATH = Path(os.getenv("SAFEMINE_MODEL", MODEL_DIR / "best.pt"))
FALLBACK_MODEL = os.getenv("SAFEMINE_FALLBACK_MODEL", "yolo11n.pt")

CONF_THRESHOLD = float(os.getenv("SAFEMINE_CONF", 0.35))   # detection confidence
IOU_THRESHOLD = 0.5                                         # NMS IoU
IMG_SIZE = 640
TRACKER_CFG = "bytetrack.yaml"                              # built into ultralytics

# Process every N-th frame (1 = every frame). 2 halves CPU time for 25-30 fps video.
FRAME_STRIDE = int(os.getenv("SAFEMINE_STRIDE", 2))
MAX_VIDEO_SECONDS = 300                                     # safety limit for demo

# --------------------------------------------------------------------------
# Unsafe-behaviour rule parameters (see docs/risk_methodology.md)
# --------------------------------------------------------------------------
# A violation must persist for at least this long before it becomes an
# "event" (temporal filter that suppresses single-frame false positives).
MIN_EVENT_SECONDS = 0.5
# Gap (seconds) that may separate two detections of the same violation
# before the episode is closed.
EVENT_GAP_SECONDS = 1.0
# A PPE-violation box is assigned to a worker when this fraction of its area
# lies inside the worker's box (intersection-over-area, IoA).
PPE_ASSIGN_IOA = 0.5
# Machinery proximity: distance from the worker's foot point to the machine
# box, normalised by the worker's box height (scale-invariant).  < 1.0 means
# the worker is closer than roughly one body height to the machine.
PROXIMITY_RATIO = 1.0

# --------------------------------------------------------------------------
# Class-name mapping (model class name -> internal role).  Lower-case keys.
# --------------------------------------------------------------------------
PERSON_CLASSES = {"person", "worker"}
NO_HELMET_CLASSES = {"no-hardhat", "no_hardhat", "no-helmet", "no_helmet", "head", "no hardhat"}
HELMET_CLASSES = {"hardhat", "helmet", "hard-hat"}
NO_VEST_CLASSES = {"no-safety vest", "no_safety_vest", "no-vest", "no_vest", "no-safety-vest", "no safety vest"}
VEST_CLASSES = {"safety vest", "safety_vest", "vest", "safety-vest"}
MACHINE_CLASSES = {"machinery", "vehicle", "truck", "bus", "car", "excavator",
                   "dump truck", "loader", "train", "forklift"}

# --------------------------------------------------------------------------
# Unsafe-behaviour categories used in the thesis
# --------------------------------------------------------------------------
CATEGORIES = {
    "NO_HELMET": "Improper PPE use - no safety helmet",
    "NO_VEST": "Improper PPE use - no high-visibility vest",
    "RESTRICTED_ZONE": "Entry into restricted / hazardous area",
    "MACHINE_PROXIMITY": "Unsafe interaction with machinery (proximity)",
}
