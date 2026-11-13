# Data

Raw sources and generated outputs stay local, except for the selected public Elo snapshots. This documentation, the fixture template, public draw snapshots, Elo snapshots, and directory placeholders are tracked by Git.

## Layout

| Directory | Contents |
| --- | --- |
| `raw/bwf_matches/` | Canonical training input: raw BWF tournament JSON |
| `draws/` | Tracked official draw snapshots used for forecasts, never training input |
| `leaderboards/` | Tracked per-event player and doubles pair Elo CSV snapshots for public viewing |
| `templates/` | `upcoming_fixtures_template.csv` for `predict.py` |
| `catalogs/` | Tracked official tournament-edition metadata and cached-source checksums |
| `odds/` | Tracked, manually transcribed sportsbook price snapshots; never training input |
| `source_cache/bwf/` | Downloaded responses, source evidence, collection reports, checksums, and original-file backups |
| `manifests/` | Local fetch/import statuses and the completed-result observation database |
| `processed/` | Generated audit summaries and future derived datasets |
| `quarantine/bwf_matches/` | Unverified files kept out of training |

## Raw Match Format

Each tournament file is a **list of rounds**, with each round a **list of match objects**:

```json
[
  [
    {
      "id": "example-match",
      "winner": 1,
      "matchTimeUtc": "2026-01-15 10:00:00",
      "duration": 45,
      "eventName": "MS",
      "roundName": "R32",
      "tournamentName": "Example Open",
      "team1": {"players": [{"id": "12345"}]},
      "team2": {"players": [{"id": "67890"}]}
    }
  ]
]
```

Player IDs must be verified BWF IDs. Doubles teams contain two players. Preserve official source fields, scores, status, tournament codes, and provenance when available.

The parser preserves source `matchTimeUtc`, falling back to unverified local `matchTime`. It separately derives `prediction_time`, `available_at`, and `availability_source`; the source start is not treated as the time a result became known. Missing individual timestamps or player IDs are not invented. A federation report with only `matchDate` remains archival rather than being assigned an invented match clock. Team-tie scores and times are not substituted for individual rubber scores or times.

Timing policy `conservative_day_history_v1` releases a result at the UTC midnight after its recorded end day when a non-midnight UTC start and a positive duration of at most 300 minutes exist. Missing durations/coarse UTC clocks use midnight two days after the source date. Unverified local dates use the earliest possible UTC date boundary (local midnight minus 14 hours) for predictions and a two-day result embargo. Only `available_at < prediction_time` results may enter historical features. These bounds assume source dates are correct; they are not verified publication times. Same-day result updates are intentionally withheld.

## Training Rules

- Para records and unplayed walkovers are excluded. Walkovers remain in raw results to determine bracket advancement; played retirements remain included.
- Duplicate IDs and fixture aliases are joined in memory, including copies with different IDs. Matching UTC copies are preferred; conflicting winners/participants exclude the whole group rather than selecting a label. Event labels and team-game slot metadata distinguish genuine repeated games. Raw files are never changed.
- Junior labels such as `BS`, `GS`, `BD`, and `GD` map to their singles/doubles disciplines. This is not a senior-only dataset. The parser retains `OTHER` for auditing, but training/new predictions accept only indoor MS, WS, MD, WD, and XD labels with valid one-/two-player teams. AirBadminton, unsupported relay/triples labels, repeated teammates, and overlapping opponents are excluded. Distinct short-game team records bearing supported event labels remain included; they are not duplicate standard matches.
- Complete normal 21-point best-of-three scores provide a winner-oriented point margin for score-sensitive Elo. Missing, retired, conflicting, incomplete, or nonstandard scores use the original Elo update. Conflicting duplicate margins are unavailable. Elo, TrueSkill, form and H2H update only when their result becomes eligible.
- Features are ordered by prediction cutoff. Result batches follow eligibility time and recorded play order, with deterministic tie-breaks. Year/early-stopping boundaries purge unavailable labels. Training never rewrites raw files.

`uv run python audit_data.py` writes `processed/data_audit.json` with coverage and exclusion counts. `--include-para` opts into para records; `--no-deduplicate` retains source duplicates for inspection.

## Official Categories

`catalogs/bwf_tournaments.csv` records official edition IDs/codes, names, calendar dates, categories, status, tournament-page URLs, and the path/checksum of the saved official calendar response. It is metadata, not match results. Empty categories stay empty; conflicting nonempty categories/codes for the same ID stop a rebuild. Response-file mtime only selects among saved snapshots, never supplies a result-publication time.

New training uses policy `official_world_tour_v1`. Only these categories override the existing numeric feature/Elo tiers:

| Official Category | Internal Model Tier |
| --- | ---: |
| World Tour Finals | 2 |
| World Tour Super 1000 | 2 |
| World Tour Super 750 | 3 |
| World Tour Super 500 | 4 |
| World Tour Super 300 | 5 |

These numbers preserve the model's established scale; they are not BWF grade/level numbers. BWF Tour Super 100, junior, continental, and other categories deliberately retain their existing heuristic in this scoped change. Missing catalog entries also retain that fallback during training and are not silently classified as World Tour events.

Training saves `tournament_catalog.csv` in the run and embeds the same snapshot in feature-state version 6. Its checksum is part of the model/calibrator policy. Catalog edits therefore cannot silently change a previously saved model. Older version 1-5 states remain usable without adopting the new mapping. Model weights, history and calibrators must still belong to the same run.

For future fixtures, add `tournament_code` (official UUID or numeric calendar ID) or `tournament_id`. Matching World Tour editions resolve from the saved catalog. Uncatalogued editions require an explicit internal `tournament_tier`; do not identify editions from names or reuse a category across years without verification. The fixture template includes an optional code column. Known IDs override stale manually supplied tiers only for catalog-aware models.

### Coverage Audit

```bash
uv run python audit_data.py --world-tour --output data/processed/world_tour_audit.json
```

The JSON and accompanying `_tournaments.csv`, `_rounds.csv`, `_gaps.csv`, and Markdown summary distinguish cached-source agreement, genuinely absent cached played records, unusable training data, cancellations/postponements, and ongoing tournaments. Walkovers and explicit `No Match` finals are accounted for without inventing played outcomes. Expected counts come from compatible saved source records, not an assumed 155-match quota.

No network fetch or raw-data rewrite is performed. `--refresh-catalog` rebuilds only from saved `api-vue-tournaments-search-*` responses; use normal collection commands to obtain newer sources separately. `--as-of <timestamp with offset>` fixes eligibility/status for a repeatable audit. Saved-source agreement does not establish that every played match was published, and the audit does not recreate which files existed at a historical cutoff.

## Fetching And Importing

```bash
uv run --extra fetch python fetch_bwf.py discover --end-date 2026-10-21
uv run python fetch_bwf.py fetch
uv run python fetch_bwf.py import-file ~/Downloads/tournament-results.json
uv run python fetch_bwf.py validate
uv run python fetch_bwf.py list-blocked
```

`backfill --start-date YYYY-MM-DD --end-date YYYY-MM-DD` collects the selected calendar range. The default start is January 1 of the earliest local year: a recent match does not establish complete historical coverage. `--refresh` explicitly re-downloads completed-source caches.

`recover-gaps --report data/source_cache/bwf/backfill-<start>-<end>/report.json` retries empty entries, checking major senior tournaments first. See each command's `--help` for transport and path options.

The standard transport uses Python HTTP; the optional `chrome` transport uses Chrome-compatible TLS/HTTP fingerprints. It does not execute challenge JavaScript or solve CAPTCHAs. Remaining access challenges and HTTP 401/403/429 responses are marked blocked.

Manual import accepts compatible JSON, CSV, or HTML containing literal JSON script data. It does not scrape arbitrary tables or execute JavaScript. Existing raw files require explicit `--force` for manual replacement; backfills merge by match ID and back up originals before atomic replacement. Thinner responses do not erase stored finished results.

Collection runs preserve `report.json`, `summary.md`, `changes.csv`, source responses, baseline checksums, and backups under `source_cache/bwf/`. Gap investigations distinguish training-ready matches, scored records lacking times, team summaries, unscored draw slots, and unresolved sources. Empty responses do not establish cancellation or complete coverage. Some older tournaments remain unresolved.

## Result Observations

Fresh successful JSON downloads record completed, supported non-para played results in `manifests/bwf_result_observations.sqlite3`. This ignored local SQLite database needs no new dependency. Inspect its counts with:

```bash
uv run python fetch_bwf.py observations
```

- `responses` records the UTC time this fetcher received a result response, its URL, and its SHA-256 checksum.
- `result_versions` stores compact result snapshots and each version's earliest `first_seen_at`. Winner, score, participants, timing, and model-relevant metadata corrections create new versions; display-name/avatar/live-serve changes do not.
- `response_results` links each receipt to its versions. Repeated unchanged results retain their original first-seen time but get a new receipt. A correction followed by a reversion remains visible in receipt order; an old version's first-seen time alone is not its full availability history. Conflicting versions within one response are preserved, not silently resolved.
- `result_invalidations` is added on the next explicit invalidation of a tracked result. It links that response to a winner reset (`0`), walkover, withdrawal, or cancellation. No completed snapshots or earlier receipts are deleted; restoring a result starts a new availability episode. Older databases remain readable without this table.

Tracking happens only on actual JSON downloads through `fetch_url`, including fresh backfill/draw requests. Cached replays, saved HTML, and manual imports do not count. Use an existing fetch command to collect; `backfill --refresh` also re-fetches completed caches. No polling schedule is installed. A logging failure stops that fetch before replacing its response cache.

**First seen is not match end or official publication time.** A six-hour fetch interval may observe a result up to roughly six hours after it becomes accessible, or longer after an outage. Fetching an old result today does not prove when it was available historically. Tracked old results use their actual receipt times, not invented historical publication times; untracked history retains conservative assumptions. This can withhold old backfilled results from historical training folds. The log is not a full live-state archive: omissions, absent/null winner fields, unknown statuses, and changes between polls cannot prove retractions. Played retirements remain valid results.

Future training defaults to `observed_results_with_conservative_history_v2`, reading `manifests/bwf_result_observations.sqlite3` and saving `observed_result_receipts.json` with the run. A missing default database means conservative fallback for all history; a missing explicitly supplied custom database is an error. `train.py --conservative-history` explicitly reproduces conservative-only timing. Training defaults to a new timestamped `outputs/runs/` directory and cannot write observed runs into active `outputs/artifacts/`. Training reads the database without changing it and never edits raw files.

Only a matching raw result fingerprint can use its receipt time. Receipts must precede the snapshot strictly; tracked IDs not yet observed, invalidated results, conflicting latest receipts, mismatched versions, and observations before the recorded prediction cutoff are excluded. Detected duplicate fixture groups cannot restore rejected results through an untracked alias. Matching observed copies supply the group's feature metadata and receipt time. Repeated versions retain their current availability episode; A-to-B-to-A corrections cannot reuse A's old first-seen time. Observed recent-form histories remain ordered by played date, not fetch order.

Saved conservative and observed-v1 models, public benchmark figures, and frozen forecasts remain unchanged. New v2 runs do not automatically replace active artifacts. These safeguards do not establish an accuracy gain or certify every historical timestamp. A useful accuracy comparison needs forecasts recorded before play and the same future matches scored without tuning on their outcomes. Back up the database with the other local data; Git does not contain it.

## Draws And Fixtures

`draws/french-open-2026.json` is the preserved pre-event draw. `draws/french-open-2026-after-r32.json` is the October 21 update with 16 confirmed survivors per event. The local `raw/bwf_matches/yonex-french-open-2026.json` has since been refreshed through the finals, with 152 played matches; the tracked draw snapshots remain historical.

Snapshots preserve bracket order, player IDs, seeds, source URLs, retrieval time, source checksums, and ordered round slots. `time_estimated` distinguishes published individual times from forecast-only session estimates. Those estimates never enter raw training data. The viewer supports complete power-of-two knockout draws and rejects missing entrants instead of inventing byes.

The archived viewer at `archive/french_open_app.py` uses `outputs/runs/2026-11-08-elo-glicko2/` for the preserved October draw when available. Run it from the repo root with `uv run --extra app python -m streamlit run archive/french_open_app.py`. Those legacy archives retain their old start-order semantics, not the corrected timing policy. The current bundle's snapshot is November 11 and rejects those past fixtures. Draw snapshots and saved model history remain in their original locations.

For individual predictions, copy `templates/upcoming_fixtures_template.csv` and replace its illustrative rows. Required fields are `match_id`, `match_time`, `discipline`, `team1_player_ids`, and `team2_player_ids`; pipe-delimit doubles IDs. Optional fields describe the round, tournament/tier, seeds, and countries. Use UTC times after the model's history cutoff.

`draws/kumamoto-masters-japan-2026-first-round.json` is the preserved pre-event R32 fixture snapshot, not a complete bracket or training input. It preserves 80 official match IDs and opponents, plus source URLs and SHA-256 checksums for all five main draws. Original responses stay in `source_cache/bwf/kumamoto-2026/`. All 80 matches were unplayed when retrieved. Of these, 59 had no published start time; those use November 10, 00:00 UTC (09:00 JST) only as a forecast reference for rest-day features. It remains archival; the current viewer uses the R16 snapshot below.

`draws/kumamoto-masters-japan-2026-r16.json` contains all five November 12 R16 draws: 39 confirmed, unplayed matches plus one XD walkover in `excluded_matches`, without a probability or betting threshold. The schedule and five draws were cross-checked against one another; all forecast times are published UTC times. The separate model run is `outputs/runs/2026-11-11-kumamoto-r16-observed/`, with a November 11 16:07:15 UTC cutoff and 302,429 training matches, including the first two days' 97 completed results. Exact model/state hashes and six official source URLs/checksums are pinned in the snapshot. Historical raw files, default active artifacts and public Elo exports are unchanged.

`odds_app.py` uses this pinned run for all 78 player/pair selections. It computes fair American lines and the first whole-number line with strictly positive model-estimated EV from full-precision probabilities. Actual sportsbook prices are entered manually, and settlement/retirement/void rules may differ. Scheduled starts in the past are excluded from current price comparisons, not treated as live forecasts. The official tournament logo is cached locally under `source_cache/bwf/kumamoto-2026/`; the viewer remains functional if that optional image is absent. No public hosting or automatic odds feed is configured. This preserved model/snapshot retains its original name-based tier; new catalog-aware models are separate runs, not silent revisions of these forecasts.

The default active model uses `two_order_average_v1` and `conservative_day_history_v1`; the separate R16 run uses `observed_results_with_conservative_history_v1`. Both opponent orders are scored from fixed eligible history and combined into complementary probabilities. Optional calibrators act on each direction before averaging. CLI exports record generation time, model/state hashes and the history cutoff. Current November 11 snapshots reject the historical Kumamoto R32 times; previous R32 exports remain archival, not corrected predictions. Older state versions remain supported only with matching legacy bundles.

Generated forecasts, result logs, models, and original Elo exports stay under the repository's ignored `outputs/` directory, not in `data/draws/`. Only the selected public Elo copies described below are committed. See the root [README](../README.md) for training and prediction commands.

## Sportsbook Odds Snapshots

`odds/<sportsbook>_<jurisdiction>/<tournament>/<recorded-date>-<round>.csv` stores one quoted selection per row. The November 11 Bet365 Ontario batch contains 64 prices for 32 Kumamoto R16 matches across MS, WS, MD and WD; no XD prices were supplied. Both sides' prices are retained, not only model-positive selections. Add a new dated/batch file for later prices instead of overwriting an existing snapshot.

The November 12 `2026-11-12-qf.csv` snapshots contain 78 prices for November 13 quarter-finals: 20 Kumamoto matches and 19 Kaohsiung matches across all five disciplines. The duplicated Japan MS screenshots are consolidated. No price was supplied for Kaohsiung's Justin HOH vs KUO Kuan Lin match. These comparisons use the unchanged active model, whose latest rating result is November 8, not a retrain on this week's results. Kaohsiung's book times differ by one hour from BWF times converted to Toronto time; both source values are preserved without assuming the book's timezone.

Rows preserve official match/player IDs and full names, the American moneyline, official UTC fixture time, literal book date/time labels, and source image names/phone clock labels. Duplicate overlapping views of the same quote are consolidated. No screenshot files are copied or committed. `recorded_at_utc` is the transcription record time, not proof of when the price was available. `captured_at_utc` stays blank because the images do not establish a full capture date, AM/PM, or timezone. Cropped book times stay blank. Book display clocks are not assumed to be UTC or substituted for official BWF times.

These are user-supplied Bet365 Ontario quotes, not independently fetched or verified live prices. In particular, the two WD Kim/Lee pairs are distinguished by their official player IDs and opponents. Preserve the short book labels separately from the resolved names to keep the mapping auditable.

Model-specific comparisons stay in ignored `outputs/odds/<tournament>/`, separate from the immutable price inputs. The November 11 comparison pins the existing R16 model/state SHA-256 hashes and full-precision probabilities. Net EV per unit staked is `p * decimal_odds - 1`; the comparison retains all positive and negative results. This does not establish profitable betting, settlement rules, closing lines, actual wagers or outcomes. Odds are not added to model features or training data.

### Paper Bet Settlement

After fetching/importing results into `raw/bwf_matches/`, run `uv run python settle_bets.py`. Use `uv run python settle_bets.py --dry-run` to preview without writing. The command reads frozen bets from `outputs/profits.json` and regenerates the human-readable `outputs/profits.md`; record future bets in the JSON ledger, not the generated Markdown.

Only normal, finished matches with matching tournament/player IDs and complete, consistent scores settle automatically. Missing/live results stay pending; retirements, walkovers and conflicting records are flagged for review rather than assumed void. Existing settled bets, original odds, probabilities, raw data and models are not changed. This is a separate post-fetch command, not a fetcher hook or scheduled task. Both ledger and report remain local and Git-ignored.

## Public Elo Snapshots

Public snapshots were rebuilt November 11 from `outputs/runs/2026-11-11-availability-safe/`: **302,332 matches through 2026-11-08 15:05 UTC**, with snapshot cutoff **2026-11-11 05:28:13 UTC**. They use the corrected deduplication/result-availability replay. Elo formulas are unchanged, but ratings and match counts can change with corrected history. These are display/download snapshots, never training input.

`leaderboards/elo_leaderboard_{MS,WS,MD,WD,XD}.csv` contains the five complete individual-player exports, totaling about 4.0 MB. Each CSV is sorted by descending `elo`. `player_id` is the BWF player identifier; `matches_in_discipline` counts included matches in that event. `last_match_time` is the most recent recorded source match time **across all events**; older unzoned clocks are not verified UTC. The snapshot eligibility cutoff itself is UTC. Doubles rows rate individual players, not pairs.

`leaderboards/elo_leaderboard_pairs_{MD,WD,XD}.csv` contains 18,619 MD, 13,653 WD, and 20,539 XD partnerships, totaling about 3.3 MB. These were generated during the same complete training replay as the individual exports. Each row contains:

- `pair_name`: both player names, falling back to IDs when a name is unavailable.
- `pair_key`: the two BWF player IDs in canonical order, separated by `|`.
- `elo`: full hybrid team Elo, `average player event Elo + n / (n + pair_lambda) * pair_elo_component`. The saved model uses `pair_lambda = 20`.
- `pair_elo_component`: the unweighted partnership-specific adjustment, not the full rating.
- `pair_matches`: their included played matches together in that event (`n`).

Pair files are sorted by descending `elo`, then `pair_key` to break rounded ties. Only two distinct players with at least one recorded played match together are included. Former partnerships remain included, and both individual ratings come from the snapshot cutoff, not the last time that pairing played. Pair ratings can therefore change when either player competes with another partner. Inactive players and junior results remain included in all snapshots; these are model ratings, not official BWF rankings.

Training now exports both individual and full pair ratings. After training, copy the eight selected exports from the run you intend to publish. For the default artifact location:

```bash
cp outputs/artifacts/elo_leaderboard_{MS,WS,MD,WD,XD}.csv data/leaderboards/
cp outputs/artifacts/elo_leaderboard_pairs_{MD,WD,XD}.csv data/leaderboards/
```

For a dated run, replace `outputs/artifacts/` with its artifact directory. Older pair exports contain only the pair component; regenerate them with the current exporter or use a new training run before publishing. Update the snapshot date, leader names/ratings in the root README, and counts here before committing. Read the eligibility cutoff from `history_as_of` and the latest recorded match from `production_refit.through`; they are different timestamps.

Commit and push the CSVs and documentation together. GitHub renders the README tables and links to the full CSV downloads; no separate website, hosting plan, or domain is required. This is a manual publishing workflow, not an automatic updater.
