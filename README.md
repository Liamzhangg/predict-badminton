# predict-badminton

LightGBM-based BWF match winner prediction with hybrid Elo ratings, Elo leaderboards, and a future-fixture scoring pipeline.

---

## Folder structure

```
predict-badminton/
├── data/
│   ├── README.md                  # Data layout and raw JSON contract
│   ├── raw/
│   │   └── bwf_matches/           # Raw BWF JSON tournament files
│   ├── processed/                 # Future cleaned/model-ready datasets
│   └── templates/
│       └── upcoming_fixtures_template.csv
├── outputs/
│   ├── artifacts/
│   │   ├── lgbm_model.pkl
│   │   ├── isotonic_calibrator.pkl
│   │   ├── platt_calibrator.pkl
│   │   ├── elo_state.pkl                # Serialised Elo engine state
│   │   ├── feature_list.json
│   │   ├── features_sample.csv
│   │   ├── feature_importance.csv
│   │   ├── metrics_report.json
│   │   ├── best_params.json             # Optuna best params (if tuned)
│   │   ├── rolling_backtest_report.json # Rolling backtest results
│   │   ├── rolling_backtest_folds.csv
│   │   ├── elo_leaderboard_MS.csv
│   │   ├── elo_leaderboard_WS.csv
│   │   ├── elo_leaderboard_MD.csv
│   │   ├── elo_leaderboard_WD.csv
│   │   ├── elo_leaderboard_XD.csv
│   │   ├── elo_leaderboard_GLOBAL.csv
│   │   ├── elo_leaderboard_pairs_MD.csv
│   │   ├── elo_leaderboard_pairs_WD.csv
│   │   └── elo_leaderboard_pairs_XD.csv
│   └── predictions/
│       └── predictions_<timestamp>.csv  # Future-fixture predictions
├── predict_badminton/
│   ├── config.py        # All constants and feature column list
│   ├── data_parser.py   # JSON → flat match records
│   ├── features.py      # Elo engine, pair Elo, feature builder, leaderboard export
│   ├── inference.py     # Feature generation for future fixtures
│   ├── model.py         # LightGBM, Optuna, CV, calibration, rolling backtest
│   └── evaluate.py      # Metrics, calibrators, reporting
├── train.py             # Main training CLI
└── predict.py           # Future fixture scoring CLI
```

---

## Training

```bash
# Full run (all data, 30 Optuna trials)
python train.py

# Skip Optuna (faster, good default params)
python train.py --no-tune

# Smoke test (100 files, no tuning, no backtest)
python train.py --smoke-test --no-tune

# Skip rolling backtest only
python train.py --no-backtest

# Custom paths
python train.py --data-dir data/raw/bwf_matches --artifacts-dir outputs/artifacts --n-trials 50
```

Training outputs:
- Model artifacts in `outputs/artifacts/`
- Elo leaderboards (per-discipline CSVs) in `outputs/artifacts/`
- Elo state snapshot at `outputs/artifacts/elo_state.pkl`
- Rolling backtest at `outputs/artifacts/rolling_backtest_report.json`

---

## Predicting future matches

### 1. Prepare a fixtures CSV

Copy the template and fill in your upcoming matches:

```bash
cp data/templates/upcoming_fixtures_template.csv data/my_fixtures.csv
```

Required columns:

| Column | Description |
|--------|-------------|
| `match_id` | Unique identifier |
| `match_time` | ISO datetime (e.g. `2026-01-15 10:00:00`) |
| `discipline` | `MS`, `WS`, `MD`, `WD`, or `XD` |
| `team1_player_ids` | Pipe-delimited player IDs (e.g. `P123` or `P123\|P456` for doubles) |
| `team2_player_ids` | Same format for team 2 |
| `round_name` | Optional: `QF`, `SF`, `Final`, etc. |
| `tournament_name` | Optional: used for tier inference |
| `tournament_tier` | Optional: 1 (elite) – 6 (lowest); defaults to 5 |
| `team1_seed` / `team2_seed` | Optional integer seeds |
| `team1_country` / `team2_country` | Optional IOC country codes |

### 2. Run inference

```bash
python predict.py --fixtures-csv data/my_fixtures.csv
```

With explicit paths:

```bash
python predict.py \
  --fixtures-csv data/my_fixtures.csv \
  --model-path outputs/artifacts/lgbm_model.pkl \
  --calibrator-path outputs/artifacts/isotonic_calibrator.pkl \
  --state-path outputs/artifacts/elo_state.pkl
```

Output: `outputs/predictions/predictions_<timestamp>.csv`

Columns: `match_id`, `match_time`, `team1_win_prob_raw`, `team1_win_prob_calibrated`, `predicted_winner`, `confidence`, plus team IDs and fixture metadata.

---

## Leaderboard output locations

After `python train.py`:

| File | Contents |
|------|----------|
| `outputs/artifacts/elo_leaderboard_MS.csv` | Men's Singles player Elo rankings |
| `outputs/artifacts/elo_leaderboard_WS.csv` | Women's Singles |
| `outputs/artifacts/elo_leaderboard_MD.csv` | Men's Doubles |
| `outputs/artifacts/elo_leaderboard_WD.csv` | Women's Doubles |
| `outputs/artifacts/elo_leaderboard_XD.csv` | Mixed Doubles |
| `outputs/artifacts/elo_leaderboard_GLOBAL.csv` | Global cross-discipline Elo |
| `outputs/artifacts/elo_leaderboard_pairs_MD.csv` | MD pair Elo components |
| `outputs/artifacts/elo_leaderboard_pairs_WD.csv` | WD pair Elo components |
| `outputs/artifacts/elo_leaderboard_pairs_XD.csv` | XD pair Elo components |

Leaderboard CSV columns: `player_id`, `elo`, `matches_in_discipline`, `last_match_time`.
Pair leaderboard columns: `pair_key` (pipe-delimited IDs), `pair_elo_component`, `pair_matches`.

---

## How pair Elo works

For doubles disciplines (MD, WD, XD) each team's rating is:

```
team_rating = avg(player_disc_elos) + w_pair(n) × pair_component
w_pair(n)   = n / (n + PAIR_LAMBDA)       # Bayesian weight-in (default λ=20)
```

`pair_component` starts at 0 and accumulates residual signal beyond individual Elos.

After each match the Elo delta is split:
- **80 %** (`ALPHA_PLAYER_SHARE`) → individual player discipline + global Elos
- **20 %** → pair component for that discipline

For singles the pair component is always 0; the six pair features are filled with 0 and are effectively ignored by the model.

All constants are configurable in `predict_badminton/config.py`:

```python
PAIR_LAMBDA: float = 20.0          # weight-in speed
ALPHA_PLAYER_SHARE: float = 0.8    # individual vs pair split
```

---

## Calibration note

Calibrators (Platt and isotonic) are fit on **out-of-fold (OOF) predictions** from walk-forward CV, not on the held-out 2025 test set. The test set is used only for final evaluation. This prevents calibration data leakage into reported test metrics.
