# Research data (Phase 1)

The web app reads `incidents.csv` (real data). If it does not exist, it falls back to
`incidents_SYNTHETIC_demo.csv`, and the Analytics page shows a yellow **SYNTHETIC** warning.
Never report synthetic numbers as findings.

## Schema (one row = one incident / observation record)
| column | type | example |
|---|---|---|
| record_id | text | MSHA-123456 |
| date | YYYY-MM-DD | 2023-05-14 |
| site_type | text | Open-pit |
| activity | text | Hauling |
| occupation | text | Haul truck driver |
| experience_years | number | 2.5 |
| unsafe_act_category | text (blank = auto-tag from narrative) | Improper PPE use |
| injury_degree | text | DAYS AWAY FROM WORK ONLY |
| severity_weight | 1-5 (blank = derived from injury_degree) | 3 |
| days_lost | number | 12 |
| shift | text | Day |
| narrative | free text | Employee was not wearing ... |

## Real data options
1. **MSHA Accident Injuries data set** (public, U.S. Mine Safety and Health Administration, 2000-present):
   download from https://www.msha.gov/mine-data-retrieval-system, then
   `python scripts/prepare_msha.py --src Accidents.txt --years 2019 2024 --surface-only`
2. Your own site records / observation checklists (Behaviour-Based Safety cards) in the same schema.
3. Questionnaire data for EGT parameter estimation (payoff perceptions) - optional, Phase 1 extension.
