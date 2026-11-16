# predict-badminton

BWF match-win probabilities using **LightGBM + score-sensitive Elo + player-based TrueSkill + country Elo, identity and counts**. Covers men's/women's singles, men's/women's doubles, and mixed doubles.

<table align="center">
  <tr>
    <th>Accuracy</th>
    <th>Log Loss</th>
    <th>Brier Score</th>
  </tr>
  <tr>
    <td align="center"><strong>77.73%</strong></td>
    <td align="center"><strong>0.4554</strong></td>
    <td align="center"><strong>0.1503</strong></td>
  </tr>
</table>

Cleaned-recipe retrospective 2026 quarterly replay: **44,188 matches**, averaged across five seeds, across all five disciplines. Lower log loss and Brier score are better. The matched pre-cleanup recipe scored 77.739% versus 77.727% cleaned, with no demonstrated performance gain. This is a data-quality promotion, not a fresh forward test or a profitability guarantee. See the [comparison report](outputs/experiments/receipt-quality-retrain-2026-11-16/report.md) and [experiment evidence](EXPERIMENTS.md).

## Forecast Default

New forecasts select `outputs/runs/2026-11-16-receipt-quality-cleaned` through `outputs/forecast_bundle.json`. The 55-feature cleaned country model uses verified final-result receipts and excludes the seven quarantined winner/score contradictions. The previous country bundle remains intact as the immediate rollback target in `outputs/forecast_bundle.previous.json`; the older observed-timing backup is also retained. Ratings and recent form refresh from completed local results before each new forecast; existing predictions and paper bets are never recalculated. Country fixtures require `team1_country` and `team2_country` columns, with empty values allowed for unknown countries. See [forecast and rollback instructions](data/README.md#new-forecasts).

## Publication Privacy

Betting slips, sportsbook odds, individual forecasts and the accounting ledger are private, local-only files. Model code and aggregate research reports are published. See the [model and data audit](outputs/audits/model-data-2026-11-16/report.md) and [publication policy](PUBLICATION.md).

Raw result data, response caches and trained model binaries also remain local; cloning the repository alone does not supply a trained prediction environment.

## Elo Leaderboards

**Rebuilt November 11, 2026; results through November 8, 15:05 UTC.** Linked names open the full CSV for that event, not just the leader shown here.

<table align="center">
  <tr>
    <th>Event</th>
    <th>Individual Leader (Elo)</th>
    <th>Pair Leader (Elo)</th>
  </tr>
  <tr>
    <td>MS</td>
    <td><a href="data/leaderboards/elo_leaderboard_MS.csv">Alex LANIER (2372.51)</a></td>
    <td>-</td>
  </tr>
  <tr>
    <td>WS</td>
    <td><a href="data/leaderboards/elo_leaderboard_WS.csv">AN Se Young (2798.83)</a></td>
    <td>-</td>
  </tr>
  <tr>
    <td>MD</td>
    <td><a href="data/leaderboards/elo_leaderboard_MD.csv">SEO Seung Jae (2166.14)</a></td>
    <td><a href="data/leaderboards/elo_leaderboard_pairs_MD.csv">KIM Won Ho / SEO Seung Jae (2199.04)</a></td>
  </tr>
  <tr>
    <td>WD</td>
    <td><a href="data/leaderboards/elo_leaderboard_WD.csv">BAEK Ha Na (2155.00)</a></td>
    <td><a href="data/leaderboards/elo_leaderboard_pairs_WD.csv">TAN Ning / LIU Sheng Shu (2289.67)</a></td>
  </tr>
  <tr>
    <td>XD</td>
    <td><a href="data/leaderboards/elo_leaderboard_XD.csv">HUANG Dong Ping (2128.83)</a></td>
    <td><a href="data/leaderboards/elo_leaderboard_pairs_XD.csv">HUANG Ya Qiong / ZHENG Si Wei (2291.71)</a></td>
  </tr>
</table>

These are **model ratings, not official BWF rankings or win probabilities**. Junior results, inactive players, and former partnerships remain included. Coverage is incomplete and snapshots are not live. See [CSV formats and refresh instructions](data/README.md#public-elo-snapshots).

## Model

Refitted on **302,934 cleaned matches spanning 2018-2026**, the default model estimates win probabilities for all five badminton disciplines.

**LightGBM** combines **score-sensitive Elo** and **player-based TrueSkill** with:

- **Player strength:** individual ratings, doubles partnerships, and past match scores.
- **Match history:** recent form, head-to-head results, and rest days.
- **Tournament context:** event type, tournament level, round, and seeding.
- **Country context:** per-discipline country Elo, country identity, and historical intercountry match counts. Home advantage and travel distance are not included.

Historical evaluation follows chronological order, with result-availability checks designed to reduce data leakage.
