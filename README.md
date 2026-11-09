# predict-badminton

BWF match-win probabilities using **LightGBM + score-sensitive Elo + player-based TrueSkill**. Covers men's/women's singles, men's/women's doubles, and mixed doubles.

## Elo Leaderboards

**Snapshot: November 8, 2026, 15:05 UTC.** Linked names open the full CSV for that event, not just the leader shown here.

| Event | Individual Leader (Elo) | Pair Leader (Elo) |
| --- | --- | --- |
| MS | [Alex LANIER (2372.75)](data/leaderboards/elo_leaderboard_MS.csv) | - |
| WS | [AN Se Young (2799.26)](data/leaderboards/elo_leaderboard_WS.csv) | - |
| MD | [SEO Seung Jae (2166.28)](data/leaderboards/elo_leaderboard_MD.csv) | [KIM Won Ho / SEO Seung Jae (2199.18)](data/leaderboards/elo_leaderboard_pairs_MD.csv) |
| WD | [BAEK Ha Na (2155.24)](data/leaderboards/elo_leaderboard_WD.csv) | [TAN Ning / LIU Sheng Shu (2290.05)](data/leaderboards/elo_leaderboard_pairs_WD.csv) |
| XD | [HUANG Dong Ping (2128.92)](data/leaderboards/elo_leaderboard_XD.csv) | [HUANG Ya Qiong / ZHENG Si Wei (2291.91)](data/leaderboards/elo_leaderboard_pairs_XD.csv) |

These are **model Elo ratings, not official BWF rankings or win probabilities**. Pair Elo includes both players' ratings plus a history-weighted partnership component. Junior results, inactive players, and former partnerships remain included. Coverage is incomplete; snapshots are manually published, not live. See [formats and refresh instructions](data/README.md#public-elo-snapshots).

## Model

The current model was retrained on **302,984 played matches from the available 2018-2026 data**, through November 8, 2026. It uses 44 features and 464 LightGBM trees:

- **Score-sensitive Elo:** individual/global strength and doubles partnership history. Complete normal scores scale updates from 0.5x to 1.5x; missing, retired, or invalid scores retain the normal update.
- **Player-based TrueSkill:** skill and uncertainty per discipline, carried between doubles partners.
- **Recent history:** win rates, match counts, rest days, and head-to-head results.
- **Context:** discipline, tournament tier, round, seeds, and country overlap.

LightGBM learns how to combine these inputs; it is not a fixed average of rating probabilities. Features are recorded before each result updates history. Training excludes para, AirBadminton, unplayed walkovers, and invalid team formats; standard junior events remain included.

### Evaluation

Same 43,580 played 2026 matches, seed 42; evaluation classifiers train only on pre-2026 data:

| Approach | Accuracy | Log Loss (Lower Is Better) |
| --- | ---: | ---: |
| Score-sensitive Elo + Glicko-2 | 75.91% | 0.483699 |
| **Score-sensitive Elo + player-based TrueSkill** | **76.51%** | **0.475303** |

Accuracy measures correct winner picks; log loss measures probability quality. Five paired seeds also favored player-based TrueSkill. The 2026 benchmark has been reused for experiments, so fresh later matches are still needed for confirmation. These metrics describe the evaluation model **before** refitting all known results, not an independent test of the production refit.

**Calendar-based inactivity uncertainty was tested but not adopted:** its small overall gain came with doubles and long-returner regressions. Rest-day inputs remain; no automatic Elo decay or extra calendar uncertainty rule is active. Detailed experiments stay local in `outputs/experiments/`; historical notes are preserved in `outputs/archives/`.

Rating parameters live in [config.py](predict_badminton/config.py), rating formulas in [features.py](predict_badminton/features.py) and [trueskill.py](predict_badminton/trueskill.py), and tree settings in [model.py](predict_badminton/model.py). The pinned `glicko2-py` dependency and `glicko.py` remain for archived model compatibility.

## Setup

Run commands from this repository's root. Python **3.12.13** is pinned in `.python-version`; `uv` manages `.venv` and dependencies through `pyproject.toml` and `uv.lock`.

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

The fetcher caches official/public responses, merges match IDs, and backs up changed raw files. It does not guarantee complete coverage. Access challenges, rate limits, and missing timestamps/player IDs remain explicit gaps; missing facts are not invented.

Use `uv run python fetch_bwf.py --help` for discovery, fetching individual URLs, validation, blocked-source listings, and gap recovery. See [data/README.md](data/README.md) for formats, import options, and storage rules.

## Train

```bash
# Evaluate, then refit a production model on all available played matches
uv run python train.py --no-tune --no-backtest --refit-all

# Small pipeline check, not a production model
uv run python train.py --smoke-test --no-tune --artifacts-dir outputs/smoke-test
```

The latest year is held out. Tree count selection uses an earlier training tail; calibrators use walk-forward out-of-fold predictions, never final holdout labels. `--refit-all` preserves the evaluation classifier as `holdout_lgbm_model.pkl`, then refits production on all results.

Models, calibrators, complete feature history, metrics, and Elo exports go to `outputs/artifacts/`. Use `--artifacts-dir outputs/runs/<name>` to preserve a separate run. Omit `--no-tune` for Optuna or `--no-backtest` for rolling backtests.

## Predict Fixtures

```bash
cp data/templates/upcoming_fixtures_template.csv data/my_fixtures.csv
# Replace the example rows with real upcoming fixtures
uv run python predict.py --fixtures-csv data/my_fixtures.csv
```

Required columns: `match_id`, `match_time`, `discipline`, `team1_player_ids`, `team2_player_ids`. Use `MS`, `WS`, `MD`, `WD`, or `XD`; separate doubles player IDs with `|`. Optional round, tournament, tier, seed, and country columns are in the template.

Times are UTC; supplied timezone offsets are converted to UTC. Every fixture must be later than the saved history cutoff. Replace the template's illustrative dates and IDs before scoring.

Training and prediction use `outputs/artifacts/` by default. For another run, override `--model-path` and `--state-path` together. Model, history, and calibrator policies must match; complete older bundles remain supported. Only load trusted local pickle files. The optional `app.py` viewer uses matching archived history for its historical draw.

CSV output goes to `outputs/predictions/`. `team1_win_prob` and `predicted_winner` use **raw probabilities by default**. Add `--probabilities platt` or `--probabilities isotonic` to use calibration; its file is loaded from the model's folder unless `--calibrator-path` overrides it. `team1_win_prob_raw` is always included; `team1_win_prob_calibrated` is included only when calibration is requested. CLI `confidence` is the scaled probability margin, `2 * abs(p - 0.5)`, not the win probability itself.

## Project Map

| Path | Purpose |
| --- | --- |
| Root Python scripts | Train, predict, fetch, audit, and optional draw viewer |
| `predict_badminton/` | Ratings, parsing, features, fitting, evaluation, and inference |
| `predict_badminton/data_sources/` | Official/public BWF collection and manual import |
| `data/` | [Formats and layout](data/README.md), raw sources, public Elo CSVs, draws, and fixture template |
| `outputs/` | Ignored local models, predictions, and experiments |
| `tests/` | Regression tests for the data pipeline, rating math, saved state, and predictions |

## Verify

```bash
uv run python -m unittest discover -s tests
uv lock --check
git diff --check
```

Git tracks code, tests, configuration, documentation, templates, draw snapshots, and selected Elo CSVs. `.venv/`, `.cache/`, raw sources, experiment outputs, model artifacts, and secrets stay ignored. These generated/local directories account for most files on disk. Back up data and trained runs separately; pushing the repo does not back them up.
