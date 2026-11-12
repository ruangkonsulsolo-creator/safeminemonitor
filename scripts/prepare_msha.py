"""
Convert the public MSHA "Accident Injuries" data set into the project schema.

1. Download the Accidents data file from the MSHA Mine Data Retrieval System
   (https://www.msha.gov/mine-data-retrieval-system -> "Explore MSHA Datasets"
   -> Accident Injuries).  It is a pipe-delimited text file (Accidents.txt)
   with a data dictionary (Accidents_Definition_File.txt).
2. Run:
     python scripts/prepare_msha.py --src path/to/Accidents.txt --years 2019 2024 --surface-only
3. Output: data/research/incidents.csv  (used automatically by the web app)

Column names follow the MSHA definition file; missing columns are tolerated.
Unsafe-act categories are tagged from the NARRATIVE field with transparent
keyword rules (backend/app/services/analytics.py: KEYWORDS) - report the
rules and a manual check of a random sample (e.g. 100 records) in the thesis.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.analytics import tag_narrative, severity_from_degree  # noqa: E402

OUT = ROOT / "data" / "research" / "incidents.csv"


def main(src, y0, y1, surface_only, sample):
    df = pd.read_csv(src, sep="|", encoding="latin-1", low_memory=False, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    print("raw rows:", len(df))
    if "CAL_YR" in df:
        yr = pd.to_numeric(df["CAL_YR"], errors="coerce")
        df = df[(yr >= y0) & (yr <= y1)]
    if surface_only and "SUBUNIT" in df:
        df = df[df["SUBUNIT"].str.contains("STRIP|QUARY|QUARRY|OPEN PIT|SURFACE", case=False, na=False)]
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    g = lambda c: df[c] if c in df else pd.Series([None] * len(df), index=df.index)  # noqa: E731
    out = pd.DataFrame({
        "record_id": g("DOCUMENT_NO"),
        "date": pd.to_datetime(g("ACCIDENT_DT"), errors="coerce").dt.date,
        "site_type": g("SUBUNIT"),
        "activity": g("ACTIVITY"),
        "occupation": g("OCCUPATION"),
        "experience_years": pd.to_numeric(g("TOT_EXPER"), errors="coerce"),
        "unsafe_act_category": g("NARRATIVE").map(tag_narrative),
        "injury_degree": g("DEGREE_INJURY"),
        "severity_weight": g("DEGREE_INJURY").map(severity_from_degree),
        "days_lost": pd.to_numeric(g("DAYS_LOST"), errors="coerce").fillna(0),
        "shift": None,
        "narrative": g("NARRATIVE"),
        "accident_type": g("ACCIDENT_TYPE"),
        "classification": g("CLASSIFICATION"),
        "injury_source": g("INJURY_SOURCE"),
    })
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")
    print(out["unsafe_act_category"].value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2024])
    ap.add_argument("--surface-only", action="store_true", help="keep surface / open-pit / quarry records")
    ap.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    a = ap.parse_args()
    main(a.src, a.years[0], a.years[1], a.surface_only, a.sample)
