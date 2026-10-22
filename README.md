# predict-badminton

BWF match predictions using LightGBM, player/pair Elo, recent form, head-to-head history, and match context. Includes data collection, training, fixture scoring, and a small Streamlit draw viewer.

## Setup

Run commands from this repository's root. Python **3.12.13** is pinned in `.python-version`; `uv` manages the local `.venv` and installs dependencies from `uv.lock`.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --extra app --extra fetch
uv run python -c "import lightgbm; print('lightgbm ok')"
```

On macOS, LightGBM needs the system OpenMP runtime: `brew install libomp` if the import reports a missing `libomp.dylib`. No separate `pip install` or environment activation is needed.

Optional extras: `app` installs Streamlit; `fetch` installs the Chrome-compatible HTTP transport; `benchmark` adds the XGBoost comparison used during training. Use `uv sync --all-extras` to install all three.

## Data

**Raw data and trained models are local, not included in a Git clone.** Put existing tournament JSON files in `data/raw/bwf_matches/`, or collect/import them:

```bash
# Collect a chosen date range; para tournaments are excluded
uv run --extra fetch python fetch_bwf.py backfill --start-date 2026-10-20 --end-date 2026-10-21

# Import a manually saved official export instead
uv run python fetch_bwf.py import-file ~/Downloads/tournament-results.json

# Check coverage and exclusions without changing raw files
uv run python audit_data.py
```

The fetcher caches official/public responses, merges results by match ID, and backs up changed raw files. It does not guarantee complete tournament coverage. Challenges, rate limits, missing timestamps, and missing player IDs remain explicit gaps; missing facts are not invented.

Use `uv run python fetch_bwf.py --help` for discovery, fetching individual URLs, validation, blocked-source listings, and gap recovery. See [data/README.md](data/README.md) for formats, import options, and storage rules.

## Train

```bash
# Evaluate, then refit a production model on all available played matches
uv run python train.py --no-tune --no-backtest --refit-all

# Small pipeline check, not a production model
uv run python train.py --smoke-test --no-tune
```

The latest data year is held out for evaluation. Tree count selection uses an earlier training tail; Platt and isotonic calibration use walk-forward out-of-fold predictions, never training-set predictions or final holdout labels.

`--refit-all` saves the evaluation model as `holdout_lgbm_model.pkl`, then fits the production model on all known results with the selected tree count. Reported test metrics describe the **pre-refit evaluation model**, not an independent test of the production refit.

Omit `--no-tune` to run Optuna; omit `--no-backtest` to run rolling backtests. Use `--artifacts-dir outputs/runs/<run-name>` to preserve a dated run rather than replace the default artifacts.

Outputs default to `outputs/artifacts/`:

| Output | Contents |
| --- | --- |
| `lgbm_model.pkl`, `*_calibrator.pkl` | Model and probability calibrators |
| `feature_state.pkl`, `feature_list.json` | Complete Elo/form/H2H history and model feature order |
| `metrics_report.json`, `feature_importance.csv` | Evaluation and feature importance |
| `elo_leaderboard_MS.csv` through `elo_leaderboard_XD.csv` | Per-event player ratings, names, IDs, counts, and last match time |
| `elo_leaderboard_GLOBAL.csv`, `elo_leaderboard_pairs_*.csv` | Global player ratings and doubles pair components |

Elo ratings are model ratings, not official BWF ranking points. Doubles strength combines average player Elo with a history-weighted pair component; a pair component alone is not its full team rating.

## Predict Fixtures

```bash
cp data/templates/upcoming_fixtures_template.csv data/my_fixtures.csv
# Replace the example rows with real upcoming fixtures
uv run python predict.py --fixtures-csv data/my_fixtures.csv
```

Required columns: `match_id`, `match_time`, `discipline`, `team1_player_ids`, `team2_player_ids`. Use `MS`, `WS`, `MD`, `WD`, or `XD`; separate doubles player IDs with `|`. Optional round, tournament, tier, seed, and country columns are in the template.

Times are UTC; supplied timezone offsets are converted to UTC. Every fixture must be later than the saved history cutoff. Replace the template's illustrative dates and IDs before scoring.

The CLI uses `outputs/artifacts/` by default. Override `--model-path`, `--calibrator-path`, and `--state-path` together when using a dated bundle. Only load trusted local pickle files.

CSV output goes to `outputs/predictions/` with raw and calibrated win probabilities. Winners use the selected calibrator, isotonic by default. CLI `confidence` is the scaled probability margin, `2 * abs(p - 0.5)`, not the win probability itself.

## French Open Viewer

The viewer currently uses the **October 21, 2026 update**: 80 confirmed R32 outcomes, including two walkovers, with 16 surviving entries per event. It covers R16 on October 22 through the October 25 finals.

Input: `data/draws/french-open-2026-after-r32.json`. Model: `outputs/runs/2026-10-21-french-open/`. If that local bundle is absent, train it from your raw data first:

```bash
uv run python train.py --no-tune --no-backtest --refit-all \
  --artifacts-dir outputs/runs/2026-10-21-french-open
uv run --extra app streamlit run app.py
```

Open `http://127.0.0.1:8501`; stop the app with `Ctrl+C` in its terminal. A snapshot requires model history earlier than its remaining fixtures; a model trained on later results must not score this old draw.

The app has draw, advancement-table, matchup, and CSV-download views. First-round boxes show confirmed opponents. Later boxes show possible slot winners; the displayed candidates plus "Other players" sum to 100%. Conditional head-to-head odds are separate from the chance that a matchup occurs.

Raw probabilities are the app default. Elo, form, and H2H stay fixed during hypothetical future rounds. Published match times are used when available; estimated session starts are labelled. Injuries, future fatigue, withdrawals, and later results are not automatically incorporated.

The pre-event snapshot and October 20 model remain separate from the R16 update. Saved local forecasts and result logs are under `outputs/forecasts/`. **The app is a saved forecast viewer, not a live feed or a deployed six-hour updater.**

## Project Map

| Path | Purpose |
| --- | --- |
| `train.py`, `predict.py`, `audit_data.py` | Training, fixture scoring, and read-only data audit |
| `fetch_bwf.py`, `predict_badminton/data_sources/` | Official/public collection and manual import |
| `predict_badminton/data_parser.py` | Raw JSON normalization, filtering, and deduplication |
| `predict_badminton/features.py`, `model.py`, `evaluate.py` | Historical features, model fitting, and evaluation |
| `predict_badminton/inference.py`, `tournament.py`, `app.py` | Future fixtures, bracket probabilities, and Streamlit UI |
| `data/` | Local sources, tracked draw snapshots, and fixture template |
| `tests/` | Parser, fetch/import, state, time-split, and bracket regression tests |

The model uses 38 features: discipline/global Elo, doubles pair history, recent win rates and rest days, H2H, tournament tier, round, seeds, country overlap, and discipline. Features are recorded before each match updates history. Tournament tiers are inferred heuristically from names; missing coverage and sparse player history limit predictions.

## Verify

```bash
uv run python -m unittest discover -s tests
uv lock --check
git diff --check
```

Git tracks code, tests, configuration, data docs, the fixture template, and public draw snapshots. Raw JSON, caches, generated reports, model artifacts, virtual environments, and secrets stay ignored. Back up local data and trained runs separately; pushing this repository does not back them up.
