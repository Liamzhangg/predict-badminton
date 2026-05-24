# Data Directory

This directory separates local source data, future derived datasets, and reusable input templates.

## Layout

```text
data/
├── raw/
│   └── bwf_matches/
├── processed/
└── templates/
    └── upcoming_fixtures_template.csv
```

## `raw/bwf_matches/`

Raw BWF tournament JSON files live here. These files are local data inputs and are ignored by Git.

The expected raw JSON structure is:

- A top-level list of rounds.
- Each round is a list of match objects.
- Match objects include fields such as `winner`, `matchTime` or `matchTimeUtc`, `eventName`, `roundName`, `tournamentName`, `team1`, and `team2`.
- Player IDs are read from `team1.players[*].id` and `team2.players[*].id`.

The training pipeline reads this directory by default:

```bash
python train.py --data-dir data/raw/bwf_matches
```

## `processed/`

Reserved for future cleaned or model-ready datasets derived from raw data.

## `templates/`

Contains reusable input templates. Start from `templates/upcoming_fixtures_template.csv` when preparing future fixtures for `predict.py`.
