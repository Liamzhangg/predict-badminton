"""Global constants and configuration."""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RAW_JSON_DIR = REPO_ROOT / "data" / "raw_matches"
ARTIFACTS_DIR = REPO_ROOT / "outputs" / "artifacts"

SEED = 42

# Elo system
DEFAULT_ELO: float = 1500.0
K_BASE: float = 32.0

DISCIPLINES = ["MS", "WS", "MD", "WD", "XD"]
DOUBLES_DISCIPLINES: frozenset[str] = frozenset({"MD", "WD", "XD"})

# Hybrid pair Elo (doubles only)
# w_pair(n) = n / (n + PAIR_LAMBDA); controls how fast pair component gains weight
PAIR_LAMBDA: float = 20.0
# Fraction of Elo delta credited to individual players vs pair component
ALPHA_PLAYER_SHARE: float = 0.8  # (1 - 0.8) = 0.2 goes to pair component

# Round importance: lower = higher stakes
ROUND_IMPORTANCE: dict[str, int] = {
    "Final": 1,
    "SF": 2,
    "QF": 3,
    "3/4": 3,
    "R16": 4,
    "R32": 5,
    "R64": 6,
    "R128": 7,
    "Qual. SF": 8,
    "Qual. QF": 9,
    "Qual. R16": 10,
    "Qual. R32": 11,
    "Qual. R64": 12,
    "R1": 13,
    "R2": 14,
    "R3": 15,
    "R4": 16,
    "R5": 17,
}

# Walk-forward CV fold boundaries (exclusive end year for validation)
# Each entry: (first_val_year, last_val_year_inclusive)
CV_FOLD_VAL_YEARS: list[tuple[int, int]] = [
    (2021, 2021),
    (2022, 2022),
    (2023, 2023),
    (2024, 2024),
]

# Feature columns used for model training
FEATURE_COLS: list[str] = [
    # Elo
    "elo_diff_disc",
    "elo_t1_disc",
    "elo_t2_disc",
    "elo_diff_global",
    "elo_t1_global",
    "elo_t2_global",
    "elo_matches_t1_disc",
    "elo_matches_t2_disc",
    "elo_reliability",
    # Pair Elo (doubles only; 0.0 for singles — safe to include for all disciplines)
    "pair_component_t1",
    "pair_component_t2",
    "pair_component_diff",
    "pair_matches_t1",
    "pair_matches_t2",
    "pair_reliability",
    # Form
    "form5_wr_t1",
    "form10_wr_t1",
    "form5_cnt_t1",
    "form10_cnt_t1",
    "rest_days_t1",
    "form5_wr_t2",
    "form10_wr_t2",
    "form5_cnt_t2",
    "form10_cnt_t2",
    "rest_days_t2",
    "form5_wr_diff",
    "form10_wr_diff",
    # H2H
    "h2h_wins_t1",
    "h2h_wins_t2",
    "h2h_total",
    "h2h_wr_t1",
    # Context
    "tournament_tier",
    "round_importance",
    "team1_seeded",
    "team2_seeded",
    "seed_diff",
    "same_country",
    # Categorical (last so LightGBM categorical_feature indexing is easy)
    "discipline",
]

CAT_FEATURE_COLS: list[str] = ["discipline"]
