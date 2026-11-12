# predict-badminton

BWF match-win probabilities using **LightGBM + score-sensitive Elo + player-based TrueSkill**. Covers men's/women's singles, men's/women's doubles, and mixed doubles.

## Elo Leaderboards

**Rebuilt November 11, 2026; results through November 8, 15:05 UTC.** Linked names open the full CSV for that event, not just the leader shown here.

| Event | Individual Leader (Elo) | Pair Leader (Elo) |
| --- | --- | --- |
| MS | [Alex LANIER (2372.51)](data/leaderboards/elo_leaderboard_MS.csv) | - |
| WS | [AN Se Young (2798.83)](data/leaderboards/elo_leaderboard_WS.csv) | - |
| MD | [SEO Seung Jae (2166.14)](data/leaderboards/elo_leaderboard_MD.csv) | [KIM Won Ho / SEO Seung Jae (2199.04)](data/leaderboards/elo_leaderboard_pairs_MD.csv) |
| WD | [BAEK Ha Na (2155.00)](data/leaderboards/elo_leaderboard_WD.csv) | [TAN Ning / LIU Sheng Shu (2289.67)](data/leaderboards/elo_leaderboard_pairs_WD.csv) |
| XD | [HUANG Dong Ping (2128.83)](data/leaderboards/elo_leaderboard_XD.csv) | [HUANG Ya Qiong / ZHENG Si Wei (2291.71)](data/leaderboards/elo_leaderboard_pairs_XD.csv) |

These are **model ratings, not official BWF rankings or win probabilities**. Junior results, inactive players, and former partnerships remain included. Coverage is incomplete and snapshots are not live. See [CSV formats and refresh instructions](data/README.md#public-elo-snapshots).

## Model

The active model uses **302,332 played matches from 2018-2026**, through November 8, 2026, with 44 features and 500 LightGBM trees:

- **Score-sensitive Elo:** individual strength and doubles partnership history. Complete normal scores scale updates from 0.5x to 1.5x; missing, retired, or invalid scores retain the normal update.
- **Player-based TrueSkill:** skill and uncertainty per discipline, carried between doubles partners.
- **Recent history:** win rates, match counts, rest days, and head-to-head results.
- **Context:** discipline, tournament tier, round, seeds, and country overlap.

LightGBM learns how to combine these inputs. Training excludes para, AirBadminton, unplayed walkovers, and invalid team formats; standard junior events remain included. There is no automatic Elo decay or extra calendar-based TrueSkill uncertainty: that experiment was not adopted.

**History is availability-aware.** The saved active model uses `conservative_day_history_v1`. Future training defaults to `observed_results_with_conservative_history_v2`: exact observed results use receipt times, while untracked older history keeps conservative day boundaries. Only results eligible strictly before each prediction cutoff enter history. See [timing rules](data/README.md#training-rules).

New training also uses the saved [official tournament catalog](data/catalogs/bwf_tournaments.csv) for World Tour tiers, joined by edition ID/code rather than tournament name. Other categories retain the old heuristic. The catalog is frozen into each new run; existing model bundles keep their original tier behavior. See [category mapping](data/README.md#official-categories).

**Predictions average both opponent orders:** `P(A wins) = (model(A,B) + 1 - model(B,A)) / 2`. Reversing the opponents produces complementary probabilities through the shared [prediction function](predict_badminton/probabilities.py). Optional calibration acts on each direction before averaging.

### Evaluation

Holdout: 43,580 played 2026 matches, seed 42. Training uses 258,737 earlier matches after deduplication and purging 15 labels unavailable at the year boundary.

| Pipeline | Accuracy | Log Loss (Lower Is Better) | Brier Score (Lower Is Better) |
| --- | ---: | ---: | ---: |
| Archived, before availability/deduplication fixes | 76.55% | 0.473802 | 0.156994 |
| **Current conservative result-availability pipeline** | **76.22%** | **0.480229** | **0.159314** |

Accuracy counts correct winner picks; log loss and Brier score assess probabilities. These are holdout-classifier metrics **before** the production refit. The timing fix also withholds more information, so the difference does not isolate leakage's effect. The 2026 benchmark has been reused for model selection; it is not an untouched final test.

## Setup

Run commands from the repository root. Python **3.12.13** is pinned in `.python-version`; `uv` manages `.venv` using `pyproject.toml` and `uv.lock`.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --extra app --extra fetch
uv run python -c "import lightgbm; print('lightgbm ok')"
```

On macOS, LightGBM needs the system OpenMP runtime: `brew install libomp` if the import reports a missing `libomp.dylib`. No separate `pip install` or environment activation is needed.

Extras: `app` adds Streamlit, `fetch` adds the Chrome-compatible HTTP transport, and `benchmark` adds the optional XGBoost comparison. `uv sync --all-extras` includes all three.

## Collect Data

**Raw data and trained models are local, not included in a Git clone.** Put tournament JSON files in `data/raw/bwf_matches/`, or collect/import them:

```bash
# Collect a chosen date range; use --refresh to re-download cached sources
uv run --extra fetch python fetch_bwf.py backfill --start-date 2026-11-10 --end-date 2026-11-11

# Import a manually saved official export instead
uv run python fetch_bwf.py import-file ~/Downloads/tournament-results.json

# Check coverage and exclusions without changing raw files
uv run python audit_data.py

# Inspect fresh-download result receipts
uv run python fetch_bwf.py observations
```

Audit World Tour coverage using the local catalog, raw files, and cached official responses (no network requests):

```bash
uv run python audit_data.py --world-tour --output data/processed/world_tour_audit.json
```

This writes a readable Markdown summary plus tournament, event/round, and missing-record CSVs next to the JSON. Cached-source agreement is not a guarantee of complete official coverage. Add `--refresh-catalog` to rebuild metadata from saved calendar responses only.

The fetcher excludes para by default, caches source evidence, and backs up changed raw files. Access challenges and incomplete sources remain explicit gaps. Fresh JSON downloads log versioned completed results in `data/manifests/bwf_result_observations.sqlite3`; cached replays and manual imports do not invent observation times. See [data formats and timing limits](data/README.md), or run `uv run python fetch_bwf.py --help` for other commands.

## Train

```bash
# Evaluate/refit a new timestamped run; does not replace active artifacts
uv run python train.py --no-tune --no-backtest --refit-all

# Small pipeline check, not a production model
uv run python train.py --smoke-test --no-tune --artifacts-dir outputs/smoke-test
```

The latest prediction year is held out. Training boundaries purge unavailable labels; optional tuning stays inside each training fold. Calibrators use training-only walk-forward predictions, never final holdout labels. `--refit-all` saves the evaluation classifier as `holdout_lgbm_model.pkl` before refitting eligible history. `--as-of <UTC timestamp>` fixes the snapshot; otherwise it uses current UTC time.

Each run saves models, calibrators, feature history, metrics, and Elo exports under a new `outputs/runs/<UTC timestamp>/` directory unless `--artifacts-dir` selects another location. **Active predictions still use `outputs/artifacts/`; training does not automatically promote a run.** Omit `--no-tune` for Optuna or `--no-backtest` for rolling backtests.

Training automatically reads `data/manifests/bwf_result_observations.sqlite3` without modifying it. If that default database does not exist, all history uses conservative timing. Each run saves its receipt snapshot. Corrections, explicit resets/withdrawals, conflicting versions, and detected fixture aliases are handled before training; missing source entries alone are not retractions. To use another database or deliberately reproduce conservative-only timing:

```bash
uv run python train.py --observations-db path/to/receipts.sqlite3 --no-tune --no-backtest
uv run python train.py --conservative-history --no-tune --no-backtest
```

Use each run's model and history together. Existing conservative/v1 bundles retain their original policies. See [observation handling and limits](data/README.md#result-observations).

`--category-catalog <CSV>` selects another official catalog. `--legacy-tournament-tiers` deliberately reproduces name-based tiers for controlled comparisons; it does not alter the observation policy. Do not choose a model from overall accuracy alone: World Tour main-draw metrics are the relevant target evaluation, and all comparisons remain retrospective until tested on new forecasts.

## Predict Fixtures

```bash
cp data/templates/upcoming_fixtures_template.csv data/my_fixtures.csv
# Replace the example rows with real upcoming fixtures
uv run python predict.py --fixtures-csv data/my_fixtures.csv
```

Required columns: `match_id`, `match_time`, `discipline`, `team1_player_ids`, `team2_player_ids`. Use verified BWF IDs, `|` between doubles partners, and UTC times **after the saved `history_as_of` cutoff**. Optional context columns are in the template.

For catalog-aware models, supply `tournament_code` (BWF UUID or calendar ID), or the optional `tournament_id`. Known World Tour editions use the run's frozen official tier even if a stale manual tier is supplied. Uncatalogued fixtures require an explicit valid `tournament_tier`; names are not used to identify editions. Older models preserve their previous behavior.

Prediction defaults to `outputs/artifacts/`. For another run, override `--model-path` and `--state-path` together. Model/history/calibrator policies must match. Only load trusted local pickle files. CSVs in `outputs/predictions/` record probabilities, generation time, history cutoff, policies, and model/state hashes. Raw probabilities are the default; `--probabilities platt` or `isotonic` loads that run's calibrator. CLI `confidence` is distance from 50/50, not the win probability.

## Odds Viewer

```bash
uv run --extra app streamlit run odds_app.py --server.port 8502
```

The viewer uses the **fixed November 11 Kumamoto R16 forecast**, with 39 matches across all five events plus an excluded XD walkover. Its separate `outputs/runs/2026-11-11-kumamoto-r16-observed/` bundle includes 302,429 matches, including 97 first-two-day results. That local run is required; active artifacts are unchanged.

It shows win probabilities, fair American lines, and model-estimated EV for manually entered prices. Past scheduled starts are excluded from current price comparisons; downloads remain archival. Model/state hashes are pinned in the draw snapshot. There is no live results or sportsbook feed.

Saved sportsbook quotes stay separate from model-specific comparisons: [odds storage notes](data/README.md#sportsbook-odds-snapshots). Expected net return per unit staked is `probability * decimal_odds - 1`; positive model EV is not proof of a profitable bet.

## Project Map

| Path | Purpose |
| --- | --- |
| `fetch_bwf.py`, `audit_data.py` | Collect sources and inspect coverage without training |
| `train.py`, `predict.py`, `odds_app.py` | Train a run, score fixtures, and view odds comparisons |
| `predict_badminton/` | Shared production code; settings in `config.py`, ratings/features in `features.py` and `trueskill.py`, fitting/evaluation in `model.py` |
| `predict_badminton/data_sources/` | Official/public BWF collection and manual import |
| `data/` | Raw sources, receipt database, public Elo CSVs, draws, templates, and price snapshots; [layout](data/README.md#layout) |
| `outputs/artifacts/`, `outputs/runs/` | Active bundle and separate dated model runs; ignored by Git |
| `outputs/experiments/`, `outputs/archives/` | Local research and historical backups, not runtime inputs for the main workflow |
| `archive/french_open_app.py` | Historical bracket viewer; run from the repo root with `uv run --extra app python -m streamlit run archive/french_open_app.py` |
| `tests/` | Pipeline, rating, saved-state, probability, and inference regression tests |

`glicko.py` and `glicko2-py` remain necessary for archived model compatibility; they are not part of the active TrueSkill model.

## Known Limits

- Historical clocks and result publication times are imperfect. Conservative timing reduces identified leakage risks; it cannot prove every historical timestamp correct.
- The receipt log catches explicit resets, walkovers, withdrawals, and cancellations of tracked results. Missing matches, null fields, unknown status codes, or changes that occur between polls cannot establish a retraction.
- Source coverage is incomplete. World Tour tiers use official cached categories for new runs; other categories and unmatched training editions retain the name heuristic. Archived experiments are not leakage-certified benchmarks.
- The next trustworthy accuracy check is a frozen model predicting new matches before play, scored later without tuning against those outcomes. No sportsbook profitability has been established.

## Verify

```bash
uv run python -m unittest discover -s tests
uv lock --check
git diff --check
```

Git tracks code, tests, configuration, documentation, templates, public draws/Elo CSVs, and manually recorded odds. `.venv/`, `.cache/`, raw sources, models, and experiments stay local and account for most files on disk. **Back up local data, observation receipts, and saved runs separately; a Git push does not include them.**
