# Methodology notes (for Chapter 3 of the thesis)

## 1. Unsafe-behaviour categories (Phase 2)
| Code | Category (thesis wording) | Detection logic | Requires |
|---|---|---|---|
| NO_HELMET | Improper PPE use – no safety helmet | `NO-Hardhat` box inside the worker box (IoA ≥ 0.5) | custom PPE model |
| NO_VEST | Improper PPE use – no high-visibility vest | `NO-Safety Vest` box inside the worker box | custom PPE model |
| RESTRICTED_ZONE | Entry into restricted / hazardous area | worker foot point inside user-drawn polygon | person class |
| MACHINE_PROXIMITY | Unsafe interaction with machinery | distance(foot point, machine box) / worker height < 1.0; workers inside the machine box (operators) excluded | person + machinery/vehicle |

Categories from the brief that are **not** implemented yet (future work): unsafe operation and unsafe
movement / working practice (need action-recognition models trained on video datasets such as CMA; see
Alotaibi & Ma 2025 and Bonyaniakbarabadi 2025).

Temporal filter: a violation becomes an *event* only after it persists ≥ 0.5 s; gaps ≤ 1.0 s are bridged.
This suppresses one-frame false positives and gives start/end timestamps.

## 2. Risk classification (HIRARC)
Risk = Likelihood (1–5) × Severity (1–5) – DOSH Malaysia (2008) *Guidelines for HIRARC*.
Bands: 1–4 Low, 5–12 Medium, 15–25 High.

* **Severity** – fixed per category, from credible worst outcome for one worker (HIRARC Table B):
  NO_HELMET 3, NO_VEST 3, RESTRICTED_ZONE 4, MACHINE_PROXIMITY 4. To be confirmed by expert rating
  and supported by Phase-1 incident statistics (mean severity per category).
* **Likelihood** – operationalised from the video: exposure ratio E = violation duration ÷ time the
  worker is visible. E < 0.10 → 1; < 0.25 → 2; < 0.50 → 3; < 0.75 → 4; else 5; +1 if the same worker
  repeats the behaviour ≥ 3 times (habitual).
* Recommendations follow the ISO 45001:2018 hierarchy of controls (clause 8.1.2).

## 3. Behavioural risk indicators (Phase 1)
* Unsafe Behaviour Rate UBR = person-frames with ≥ 1 violation ÷ all person-frames.
* Helmet / vest compliance rate = 1 − violating person-frames ÷ person-frames.
* Events per minute of video; mean exposure ratio per category.
* Incident data: Behavioural Risk Index BRI_c = share_c × mean severity_c ÷ 5; Pareto; χ² test of
  experience band × category (report Cramér's V and % of expected counts < 5).

## 4. EGT model
See `backend/app/services/egt.py` (docstring) – payoffs are defined per pure profile, replicator
equations derived mechanically, RK4 integration (dt = 0.05, T = 50), Jacobian eigenvalues at the 8
vertices (Friedman 1991). Phase-2 link: α_I = 1 − (1 − α0)(1 − φr), β_I = 1 − (1 − β0)(1 − φr),
x0 = 1 − UBR. Parameter values are illustrative and must be justified (literature, expert survey)
before reporting; always present sensitivity analysis.
