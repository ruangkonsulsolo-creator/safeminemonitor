"""
Python-based analytics (Phase 1: computational analysis of unsafe behaviour).

Two data sources are analysed with the same code path:

  A. Research data (incident / accident / observation records)
     data/research/incidents.csv    <- real data (e.g. MSHA Accident Injuries
                                       data set prepared by scripts/prepare_msha.py,
                                       or the student's own site records)
     data/research/incidents_SYNTHETIC_demo.csv <- clearly-labelled demo data,
                                       used ONLY when no real file exists.

  B. Detection records produced by the computer-vision pipeline (SQLite).

Outputs: data-quality report, descriptive statistics, Pareto analysis,
cross-tabulation with chi-square test, temporal trend, and behavioural risk
indicators (BRI) that feed the risk module and the EGT calibration.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESEARCH_DIR, CATEGORIES

REAL_FILE = RESEARCH_DIR / "incidents.csv"
DEMO_FILE = RESEARCH_DIR / "incidents_SYNTHETIC_demo.csv"

SCHEMA = ["record_id", "date", "site_type", "activity", "occupation", "experience_years",
          "unsafe_act_category", "injury_degree", "severity_weight", "days_lost", "shift", "narrative"]

# Keyword rules for tagging free-text narratives with an unsafe-act category
# (transparent, reproducible text mining; refine with the supervisor).
KEYWORDS = {
    "Improper PPE use": [r"hard ?hat", r"helmet", r"safety glasses", r"not wearing", r"without (his|her|a) ",
                         r"glove", r"respirator", r"\bppe\b", r"harness", r"lanyard", r"hearing protection"],
    "Unsafe interaction with machinery": [r"haul truck", r"loader", r"struck by", r"run over", r"pinned",
                                          r"caught (in|between)", r"conveyor", r"dozer", r"excavator",
                                          r"forklift", r"shuttle car", r"scoop", r"mobile equipment"],
    "Entry into restricted / hazardous area": [r"high ?wall", r"\bedge\b", r"barricad", r"restricted",
                                               r"blast(ing)? area", r"ground fall", r"roof fall", r"rib fall",
                                               r"unsupported roof", r"exclusion zone"],
    "Unsafe movement / working practice": [r"slipp?ed", r"tripp?ed", r"\bfell\b", r"jump(ed)?", r"climb",
                                           r"ladder", r"lifting", r"pulling", r"pushing", r"overexert",
                                           r"strain", r"twist"],
}


def tag_narrative(text: str) -> str:
    t = (text or "").lower()
    for cat, pats in KEYWORDS.items():
        if any(re.search(p, t) for p in pats):
            return cat
    return "Other / unclassified"


def severity_from_degree(degree: str) -> int:
    d = (degree or "").upper()
    if "FATAL" in d:
        return 5
    if "PERM" in d:
        return 4
    if "AWAY" in d or "AWY" in d:
        return 3
    if "RESTR" in d or "RSTR" in d:
        if "NO" in d and ("NO DYS" in d or "NO DAYS" in d):
            return 1
        return 2
    if d.strip():
        return 1
    return np.nan


# ------------------------------------------------------------------ loading
def load_incidents(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    if path is None:
        path = REAL_FILE if REAL_FILE.exists() else DEMO_FILE
    if not Path(path).exists():
        return pd.DataFrame(columns=SCHEMA), {"source": None, "synthetic": True}
    df = pd.read_csv(path)
    meta = {"source": Path(path).name, "synthetic": "SYNTHETIC" in Path(path).name.upper()}
    return df, meta


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_raw": int(len(df))}
    df = df.copy()
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    report["missing_by_column"] = {c: int(df[c].isna().sum()) for c in SCHEMA}
    df = df.drop_duplicates(subset=["record_id"]) if df["record_id"].notna().any() else df.drop_duplicates()
    report["duplicates_removed"] = report["rows_raw"] - int(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
    df["experience_band"] = pd.cut(df["experience_years"], [-0.1, 1, 3, 10, 100],
                                   labels=["<1 y", "1-3 y", "3-10 y", ">10 y"]).astype(str)
    df["days_lost"] = pd.to_numeric(df["days_lost"], errors="coerce").fillna(0)
    sw = pd.to_numeric(df["severity_weight"], errors="coerce")
    df["severity_weight"] = sw.fillna(df["injury_degree"].map(severity_from_degree)).fillna(1)
    # narrative-based tagging when category is missing
    miss = df["unsafe_act_category"].isna() | (df["unsafe_act_category"].astype(str).str.strip() == "")
    df.loc[miss, "unsafe_act_category"] = df.loc[miss, "narrative"].map(tag_narrative)
    report["rows_clean"] = int(len(df))
    report["categories_tagged_from_text"] = int(miss.sum())
    return df, report


# ------------------------------------------------------------------ statistics
def chi_square(table: pd.DataFrame) -> dict:
    obs = table.values.astype(float)
    if obs.size == 0 or obs.sum() == 0 or min(obs.shape) < 2:
        return {"chi2": None, "dof": None, "p_value": None, "cramers_v": None}
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / obs.sum()
    chi2 = float(((obs - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, dof))
    except ImportError:  # pragma: no cover
        p = None
    v = math.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1)))
    return {"chi2": round(chi2, 3), "dof": int(dof), "p_value": p, "cramers_v": round(v, 3),
            "low_expected_cells_pct": round(float((exp < 5).mean() * 100), 1)}


def pareto(series: pd.Series) -> list[dict]:
    vc = series.value_counts()
    cum = vc.cumsum() / vc.sum() * 100
    return [{"label": str(k), "count": int(v), "cum_pct": round(float(c), 1)} for k, v, c in zip(vc.index, vc, cum)]


def incident_analysis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    cat = df["unsafe_act_category"]
    sev = df.groupby("unsafe_act_category").agg(
        n=("record_id", "size"), mean_severity=("severity_weight", "mean"),
        total_days_lost=("days_lost", "sum"), fatal=("severity_weight", lambda s: int((s >= 5).sum())))
    total = len(df)
    sev["frequency_share"] = sev["n"] / total
    # Behavioural Risk Index of a category = frequency share x mean severity (normalised 0-1)
    sev["bri"] = sev["frequency_share"] * sev["mean_severity"] / 5.0
    sev = sev.sort_values("bri", ascending=False)
    ct = pd.crosstab(df["experience_band"], cat)
    ct_occ = pd.crosstab(df["occupation"], cat)
    top_occ = ct_occ.sum(1).sort_values(ascending=False).head(8).index
    trend = df.groupby("month").size()
    return {
        "n_records": int(total),
        "period": [str(df["date"].min().date()), str(df["date"].max().date())],
        "pareto_category": pareto(cat),
        "category_indicators": [
            {"category": k, "n": int(r.n), "share": round(float(r.frequency_share), 3),
             "mean_severity": round(float(r.mean_severity), 2), "days_lost": int(r.total_days_lost),
             "fatal": int(r.fatal), "bri": round(float(r.bri), 3)} for k, r in sev.iterrows()],
        "by_occupation": {"index": [str(i) for i in top_occ], "columns": [str(c) for c in ct_occ.columns],
                          "values": ct_occ.loc[top_occ].values.tolist()},
        "experience_crosstab": {"index": [str(i) for i in ct.index], "columns": [str(c) for c in ct.columns],
                                "values": ct.values.tolist(), "chi_square": chi_square(ct)},
        "by_shift": {str(k): int(v) for k, v in df["shift"].fillna("Unknown").value_counts().items()},
        "severity_distribution": {str(int(k)): int(v) for k, v in df["severity_weight"].value_counts().sort_index().items()},
        "monthly_trend": {"labels": list(trend.index), "values": [int(v) for v in trend.values]},
    }


# ------------------------------------------------------------------ CV-based indicators
def detection_indicators(jobs: list[dict], events: list[dict]) -> dict:
    """Behavioural risk indicators computed from the vision system's records."""
    person_frames = sum(j.get("person_frames") or 0 for j in jobs)
    unsafe_pf = sum(j.get("unsafe_person_frames") or 0 for j in jobs)
    duration = sum(j.get("duration_s") or 0 for j in jobs if j.get("media_type") == "video")
    by_cat = {}
    for e in events:
        by_cat.setdefault(e["category"], []).append(e)
    helmet_pf = sum((j.get("summary") or {}).get("helmet_violation_person_frames", 0) for j in jobs)
    vest_pf = sum((j.get("summary") or {}).get("vest_violation_person_frames", 0) for j in jobs)
    ubr = unsafe_pf / person_frames if person_frames else None
    return {
        "jobs_analysed": len(jobs),
        "video_minutes": round(duration / 60, 2),
        "person_frames": person_frames,
        "unsafe_behaviour_rate": None if ubr is None else round(ubr, 4),
        "helmet_compliance_rate": None if not person_frames else round(1 - helmet_pf / person_frames, 4),
        "vest_compliance_rate": None if not person_frames else round(1 - vest_pf / person_frames, 4),
        "events_total": len(events),
        "events_per_minute": None if not duration else round(len(events) / (duration / 60), 3),
        "by_category": {
            c: {"label": CATEGORIES.get(c, c), "events": len(v),
                "mean_duration_s": round(float(np.mean([e["duration_s"] or 0 for e in v])), 2),
                "mean_exposure": round(float(np.mean([e["exposure_ratio"] or 0 for e in v])), 3),
                "high_risk": sum(1 for e in v if e.get("risk_level") == "High")}
            for c, v in by_cat.items()},
    }


def full_report(jobs: list[dict], events: list[dict], path: Path | None = None) -> dict:
    df, meta = load_incidents(path)
    clean, quality = preprocess(df) if not df.empty else (df, {"rows_raw": 0})
    return {"dataset": meta, "data_quality": quality,
            "incidents": incident_analysis(clean) if not clean.empty else {},
            "detection": detection_indicators(jobs, events)}
