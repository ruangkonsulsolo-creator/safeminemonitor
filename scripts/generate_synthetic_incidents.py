"""
Generate a SYNTHETIC demo incident dataset with the project schema.

!! For software testing and UI demonstration only. Do NOT report these numbers
!! as research findings. Replace with real data (scripts/prepare_msha.py or
!! your own site records) -> data/research/incidents.csv

Usage:  python scripts/generate_synthetic_incidents.py --n 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "research" / "incidents_SYNTHETIC_demo.csv"

CATS = {
    "Improper PPE use": (0.30, [1, 2, 3, 3, 4]),
    "Unsafe interaction with machinery": (0.22, [2, 3, 3, 4, 5]),
    "Entry into restricted / hazardous area": (0.13, [2, 3, 4, 4, 5]),
    "Unsafe movement / working practice": (0.27, [1, 2, 2, 3, 3]),
    "Other / unclassified": (0.08, [1, 1, 2, 3]),
}
OCC = ["Haul truck driver", "Loader operator", "Driller", "Blaster", "Labourer", "Mechanic",
       "Electrician", "Surveyor", "Supervisor"]
ACT = ["Hauling", "Loading", "Drilling", "Blasting preparation", "Maintenance", "Walking / travelling",
       "Material handling", "Inspection"]
DEGREE = {1: "NO DYS AWY FRM WRK,NO RSTR ACT", 2: "DAYS RESTRICTED ACTIVITY ONLY",
          3: "DAYS AWAY FROM WORK ONLY", 4: "PERM TOT OR PERM PRTL DISABLTY", 5: "FATALITY"}


def main(n, seed):
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    probs = np.array([CATS[c][0] for c in cats])
    rows = []
    start = pd.Timestamp("2023-01-01")
    for i in range(n):
        c = rng.choice(cats, p=probs / probs.sum())
        exp_years = float(np.round(rng.gamma(2.0, 3.0 if c != "Improper PPE use" else 1.8), 1))
        sev = int(rng.choice(CATS[c][1]))
        rows.append({
            "record_id": f"SYN-{i:05d}",
            "date": (start + pd.Timedelta(days=int(rng.integers(0, 730)))).date().isoformat(),
            "site_type": rng.choice(["Open-pit", "Underground", "Quarry"], p=[.6, .25, .15]),
            "activity": rng.choice(ACT), "occupation": rng.choice(OCC),
            "experience_years": exp_years, "unsafe_act_category": c,
            "injury_degree": DEGREE[sev], "severity_weight": sev,
            "days_lost": int(0 if sev <= 1 else rng.poisson({2: 3, 3: 15, 4: 60, 5: 0}[sev])),
            "shift": rng.choice(["Day", "Night"], p=[.62, .38]),
            "narrative": "SYNTHETIC RECORD - demo only",
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} synthetic rows -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sys.exit(main(a.n, a.seed))
