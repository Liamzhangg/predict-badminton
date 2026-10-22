# Data

Raw sources and generated outputs stay local, except for the selected public Elo snapshots. This documentation, the fixture template, public draw snapshots, Elo snapshots, and directory placeholders are tracked by Git.

## Layout

| Directory | Contents |
| --- | --- |
| `raw/bwf_matches/` | Canonical training input: raw BWF tournament JSON |
| `draws/` | Tracked official draw snapshots used for forecasts, never training input |
| `leaderboards/` | Tracked per-event player and doubles pair Elo CSV snapshots for public viewing |
| `templates/` | `upcoming_fixtures_template.csv` for `predict.py` |
| `source_cache/bwf/` | Downloaded responses, source evidence, collection reports, checksums, and original-file backups |
| `manifests/` | Local fetch/import statuses, including blocked sources |
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

The parser prefers `matchTimeUtc`, then falls back to `matchTime`. Missing individual timestamps or player IDs are not invented. A federation report with only a date can be archived with `matchDate`, but cannot become a time-ordered training record. Team-tie scores and times are not substituted for individual rubber scores or times.

## Training Rules

- Para records and unplayed walkovers are excluded. Walkovers remain in raw results to determine bracket advancement; played retirements remain included.
- Duplicate IDs or deterministic metadata signatures are deduplicated in memory. Source aliases prefer the UTC copy only when local result and match-slot metadata agree.
- Junior labels such as `BS`, `GS`, `BD`, and `GD` map to their standard singles/doubles disciplines. This is not a senior-only dataset. Unrecognized formats retain `OTHER`; team relay and triples are not relabelled as ordinary singles/doubles.
- All included records are sorted chronologically. Training never rewrites raw files.

`uv run python audit_data.py` writes `processed/data_audit.json` with coverage and exclusion counts. `--include-para` opts into para records; `--no-deduplicate` retains source duplicates for inspection.

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

## Draws And Fixtures

`draws/french-open-2026.json` is the preserved pre-event draw. `draws/french-open-2026-after-r32.json` is the October 21 update with 16 confirmed survivors per event. The 80 completed R32 outcomes are stored locally in `raw/bwf_matches/yonex-french-open-2026.json`; 78 were played and two were walkovers.

Snapshots preserve bracket order, player IDs, seeds, source URLs, retrieval time, source checksums, and ordered round slots. `time_estimated` distinguishes published individual times from forecast-only session estimates. Those estimates never enter raw training data. The viewer supports complete power-of-two knockout draws and rejects missing entrants instead of inventing byes.

For individual predictions, copy `templates/upcoming_fixtures_template.csv` and replace its illustrative rows. Required fields are `match_id`, `match_time`, `discipline`, `team1_player_ids`, and `team2_player_ids`; pipe-delimit doubles IDs. Optional fields describe the round, tournament/tier, seeds, and countries. Use UTC times after the model's history cutoff.

Generated forecasts, result logs, models, and original Elo exports stay under the repository's ignored `outputs/` directory, not in `data/draws/`. Only the selected public Elo copies described below are committed. See the root [README](../README.md) for training and prediction commands.

## Public Elo Snapshots

All public snapshots use model history through **2026-10-21 18:10 UTC**, from `outputs/runs/2026-10-21-french-open/`. They are display/download snapshots, never training input.

`leaderboards/elo_leaderboard_{MS,WS,MD,WD,XD}.csv` contains the five complete individual-player exports, totaling about 3.9 MB. Each CSV is sorted by descending `elo`. `player_id` is the BWF player identifier; `matches_in_discipline` counts included matches in that event. `last_match_time` is the player's most recent included match **across all events**, in UTC. Doubles rows in these files rate individual players, not pairs.

`leaderboards/elo_leaderboard_pairs_{MD,WD,XD}.csv` contains 18,489 MD, 13,599 WD, and 20,319 XD partnerships, totaling about 3.3 MB. These were regenerated from the same saved `feature_state.pkl` using the updated exporter, without retraining or altering the original run. Each row contains:

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

For a dated run, replace `outputs/artifacts/` with its artifact directory. Older pair exports contain only the pair component; regenerate them with the current exporter or use a new training run before publishing. Update the snapshot date, player/pair counts, and top-five tables in the root README to match the new exports before committing. Read the history cutoff from `production_refit.through` in that run's `metrics_report.json` when using `--refit-all`.

Commit and push the CSVs and documentation together. GitHub renders the README tables and links to the full CSV downloads; no separate website, hosting plan, or domain is required. This is a manual publishing workflow, not an automatic updater.
