# predict-badminton

BWF match predictions using **LightGBM + Elo + Glicko-2**, recent form, head-to-head history, and match context. Includes data collection, training, fixture scoring, and a small Streamlit draw viewer.

## Elo Leaderboards

**Snapshot: October 21, 2026, 18:10 UTC**, after processing 300,103 played matches from the available 2018-2026 data. These are this model's Elo ratings, **not official BWF rankings or win probabilities**. Coverage is incomplete, junior results are included, and inactive players are not filtered out or penalized for inactivity.

Each leaderboard below shows its top five, with full CSV downloads. Individual ratings and doubles pair ratings are listed separately.

### Individual Players

These tables rank players individually, including in doubles events.

<details open>
<summary>Men's Singles (MS)</summary>

| Rank | Player | Elo | Matches |
| --- | --- | ---: | ---: |
| 1 | Alex LANIER | 2353.38 | 353 |
| 2 | SHI Yu Qi | 2276.70 | 363 |
| 3 | Anders ANTONSEN | 2260.67 | 401 |
| 4 | Viktor AXELSEN | 2245.19 | 362 |
| 5 | Christo POPOV | 2221.52 | 411 |

[Full MS leaderboard: 15,949 players](data/leaderboards/elo_leaderboard_MS.csv)

</details>

<details>
<summary>Women's Singles (WS)</summary>

| Rank | Player | Elo | Matches |
| --- | --- | ---: | ---: |
| 1 | AN Se Young | 2739.22 | 488 |
| 2 | Akane YAMAGUCHI | 2463.93 | 484 |
| 3 | WANG Zhi Yi | 2404.21 | 426 |
| 4 | CHEN Yu Fei | 2365.95 | 463 |
| 5 | Carolina MARIN | 2287.18 | 272 |

[Full WS leaderboard: 10,129 players](data/leaderboards/elo_leaderboard_WS.csv)

</details>

<details>
<summary>Men's Doubles (MD)</summary>

| Rank | Player | Elo | Matches |
| --- | --- | ---: | ---: |
| 1 | SEO Seung Jae | 2147.05 | 420 |
| 2 | Muhammad Shohibul FIKRI | 2045.35 | 328 |
| 3 | KIM Won Ho | 2021.12 | 330 |
| 4 | Fajar ALFIAN | 1975.86 | 446 |
| 5 | WANG Chang | 1947.88 | 380 |

[Full MD leaderboard: 17,564 players](data/leaderboards/elo_leaderboard_MD.csv)

</details>

<details>
<summary>Women's Doubles (WD)</summary>

| Rank | Player | Elo | Matches |
| --- | --- | ---: | ---: |
| 1 | BAEK Ha Na | 2132.52 | 376 |
| 2 | LIU Sheng Shu | 2086.21 | 285 |
| 3 | JIA Yi Fan | 2025.71 | 451 |
| 4 | KIM Hye Jeong | 1993.67 | 276 |
| 5 | ZHANG Shu Xian | 1991.39 | 348 |

[Full WD leaderboard: 11,651 players](data/leaderboards/elo_leaderboard_WD.csv)

</details>

<details>
<summary>Mixed Doubles (XD)</summary>

| Rank | Player | Elo | Matches |
| --- | --- | ---: | ---: |
| 1 | HUANG Dong Ping | 2076.99 | 477 |
| 2 | HUANG Ya Qiong | 2047.66 | 335 |
| 3 | WEI Ya Xin | 2001.35 | 303 |
| 4 | FENG Yan Zhe | 1990.56 | 366 |
| 5 | ZHENG Si Wei | 1987.23 | 327 |

[Full XD leaderboard: 19,337 players](data/leaderboards/elo_leaderboard_XD.csv)

</details>

### Doubles Pairs

Pair Elo uses the model's full team rating: `average of both players' event Elo + n / (n + 20) * pair component`, where `n` is their played matches together. It is not the pair component alone.

These lists include every recorded two-player partnership with at least one played match, including former partnerships. Both players' ratings are taken at the snapshot cutoff, not frozen at their last match together. Match counts show how much shared history supports each pairing; no minimum-history or active-pair filter is applied.

<details>
<summary>Men's Doubles (MD) Pairs</summary>

| Rank | Pair | Elo | Matches Together |
| --- | --- | ---: | ---: |
| 1 | KIM Won Ho / SEO Seung Jae | 2166.68 | 149 |
| 2 | SEO Seung Jae / KANG Min Hyuk | 2150.98 | 148 |
| 3 | Fajar ALFIAN / Muhammad Shohibul FIKRI | 2084.55 | 85 |
| 4 | SEO Seung Jae / CHOI Sol Gyu | 2056.81 | 101 |
| 5 | SEO Seung Jae / JIN Yong | 2055.06 | 17 |

[Full MD pair leaderboard: 18,489 pairs](data/leaderboards/elo_leaderboard_pairs_MD.csv)

</details>

<details>
<summary>Women's Doubles (WD) Pairs</summary>

| Rank | Pair | Elo | Matches Together |
| --- | --- | ---: | ---: |
| 1 | TAN Ning / LIU Sheng Shu | 2230.01 | 262 |
| 2 | JIA Yi Fan / CHEN Qing Chen | 2171.24 | 328 |
| 3 | BAEK Ha Na / LEE So Hee | 2157.73 | 235 |
| 4 | Nami MATSUYAMA / Chiharu SHIDA | 2086.83 | 308 |
| 5 | BAEK Ha Na / KIM Hye Jeong | 2066.38 | 5 |

[Full WD pair leaderboard: 13,599 pairs](data/leaderboards/elo_leaderboard_pairs_WD.csv)

</details>

<details>
<summary>Mixed Doubles (XD) Pairs</summary>

| Rank | Pair | Elo | Matches Together |
| --- | --- | ---: | ---: |
| 1 | HUANG Ya Qiong / ZHENG Si Wei | 2252.88 | 323 |
| 2 | FENG Yan Zhe / HUANG Dong Ping | 2156.38 | 267 |
| 3 | WEI Ya Xin / JIANG Zhen Bang | 2154.38 | 269 |
| 4 | HUANG Dong Ping / WANG Yi Lyu | 2101.88 | 204 |
| 5 | Yuta WATANABE / Arisa IGARASHI | 2072.60 | 291 |

[Full XD pair leaderboard: 20,319 pairs](data/leaderboards/elo_leaderboard_pairs_XD.csv)

</details>

These are manually published snapshots, not a live feed. See [refresh instructions](data/README.md#public-elo-snapshots).

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

See [model results](MODEL_RESULTS.md) for a compact comparison of tested approaches and untested ideas.

```bash
# Evaluate, then refit a production model on all available played matches
uv run python train.py --no-tune --no-backtest --refit-all

# Small pipeline check, not a production model
uv run python train.py --smoke-test --no-tune --artifacts-dir outputs/smoke-test
```

The main model combines 38 Elo/context features with nine Glicko-2 features: each side's rating, rating deviation (uncertainty), volatility, and their differences. LightGBM learns how to combine these inputs; this is not a fixed average of Elo and Glicko probabilities.

Glicko-2 uses Monday-to-Sunday UTC periods, starting at rating 1500, deviation 350, and volatility 0.06, with `tau=0.5`. Ratings are separate for each event and exact doubles partnership; a new partnership starts fresh. Observed results are batched after each week, while Elo/form/H2H update after every match. Inactivity increases Glicko uncertainty, not Elo decay or a direct rating penalty. Future previews never add hypothetical results or mutate history.

The November 8 integration reproduced **75.79% accuracy / 0.4853 log loss** on 39,537 played 2026 matches, versus **74.80% / 0.5018** for the Elo-based control. This reused holdout is exploratory, not a guarantee on future tournaments. Raw probabilities remain the default; Platt and isotonic calibration are available for comparison. See [model results](MODEL_RESULTS.md) for all runs.

The latest data year is held out for evaluation. Tree count selection uses an earlier training tail; Platt and isotonic calibration use walk-forward out-of-fold predictions, never training-set predictions or final holdout labels.

`--refit-all` saves the evaluation model as `holdout_lgbm_model.pkl`, then fits the production model on all known results with the selected tree count. Reported test metrics describe the **pre-refit evaluation model**, not an independent test of the production refit.

Omit `--no-tune` to run Optuna; omit `--no-backtest` to run rolling backtests. Use `--artifacts-dir outputs/runs/<run-name>` to preserve a dated run rather than replace the default artifacts.

Outputs default to `outputs/artifacts/`:

| Output | Contents |
| --- | --- |
| `lgbm_model.pkl`, `*_calibrator.pkl` | Model and probability calibrators |
| `feature_state.pkl`, `feature_list.json` | Complete Elo/Glicko-2/form/H2H history, pending weekly results, and model feature order |
| `metrics_report.json`, `feature_importance.csv` | Evaluation and feature importance |
| `elo_leaderboard_MS.csv` through `elo_leaderboard_XD.csv` | Per-event player ratings, names, IDs, counts, and last match time |
| `elo_leaderboard_GLOBAL.csv` | Global player ratings |
| `elo_leaderboard_pairs_MD.csv`, `elo_leaderboard_pairs_WD.csv`, `elo_leaderboard_pairs_XD.csv` | Named doubles pair ratings, player IDs, pair components, and shared match counts |

Elo ratings are model ratings, not official BWF ranking points. Doubles strength combines average player Elo with a history-weighted pair component; a pair component alone is not its full team rating.

## Predict Fixtures

```bash
cp data/templates/upcoming_fixtures_template.csv data/my_fixtures.csv
# Replace the example rows with real upcoming fixtures
uv run python predict.py --fixtures-csv data/my_fixtures.csv
```

Required columns: `match_id`, `match_time`, `discipline`, `team1_player_ids`, `team2_player_ids`. Use `MS`, `WS`, `MD`, `WD`, or `XD`; separate doubles player IDs with `|`. Optional round, tournament, tier, seed, and country columns are in the template.

Times are UTC; supplied timezone offsets are converted to UTC. Every fixture must be later than the saved history cutoff. Replace the template's illustrative dates and IDs before scoring.

Training, the prediction CLI, and the app use `outputs/artifacts/` by default. For a dated bundle, override `--model-path` and `--state-path` together. Complete older Elo-only bundles remain supported; mixing an old model with new history is rejected. Only load trusted local pickle files.

CSV output goes to `outputs/predictions/`. `team1_win_prob` and `predicted_winner` use **raw probabilities by default**. Add `--probabilities platt` or `--probabilities isotonic` to use calibration; its file is loaded from the model's folder unless `--calibrator-path` overrides it. `team1_win_prob_raw` is always included; `team1_win_prob_calibrated` is included only when calibration is requested. CLI `confidence` is the scaled probability margin, `2 * abs(p - 0.5)`, not the win probability itself.

## Project Map

| Path | Purpose |
| --- | --- |
| `train.py`, `predict.py`, `audit_data.py` | Training, fixture scoring, and read-only data audit |
| `fetch_bwf.py`, `predict_badminton/data_sources/` | Official/public collection and manual import |
| `predict_badminton/data_parser.py` | Raw JSON normalization, filtering, and deduplication |
| `predict_badminton/features.py`, `glicko.py`, `model.py`, `evaluate.py` | Historical features, weekly ratings, model fitting, and evaluation |
| `predict_badminton/inference.py`, `tournament.py`, `app.py` | Future fixtures, bracket probabilities, and Streamlit UI |
| `data/` | Local sources, public Elo snapshots, tracked draws, and fixture template |
| `tests/` | Parser, fetch/import, state, time-split, and bracket regression tests |

The model uses 47 features: discipline/global Elo, doubles pair history, Glicko-2 ratings/uncertainty/volatility, recent win rates and rest days, H2H, tournament tier, round, seeds, country overlap, and discipline. Features are recorded before each match updates history. Tournament tiers are inferred heuristically from names; missing coverage and sparse player history limit predictions. Rating parameters live in [config.py](predict_badminton/config.py); tree parameters live in [model.py](predict_badminton/model.py). Glicko-2 math comes from pinned `glicko2-py==0.1.0` and is regression-tested against the [published example](https://www.glicko.net/glicko/glicko2.pdf).

## Verify

```bash
uv run python -m unittest discover -s tests
uv lock --check
git diff --check
```

Git tracks code, tests, configuration, data docs, the fixture template, public draw snapshots, and the selected Elo CSVs in `data/leaderboards/`. Raw JSON, caches, generated reports, model artifacts, virtual environments, and secrets stay ignored. Back up local data and trained runs separately; pushing this repository does not back them up.
