# predict-badminton

BWF match-win probabilities using **LightGBM + score-sensitive Elo + player-based TrueSkill**. Covers men's/women's singles, men's/women's doubles, and mixed doubles.

<table align="center">
  <tr>
    <th>Accuracy</th>
    <th>Log Loss</th>
    <th>Brier Score</th>
  </tr>
  <tr>
    <td align="center"><strong>76.22%</strong></td>
    <td align="center"><strong>0.4802</strong></td>
    <td align="center"><strong>0.1593</strong></td>
  </tr>
</table>

Retrospective 2026 holdout: **43,580 matches** across all five disciplines. Lower log loss and Brier score are better.

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

Built from **302,332 matches spanning 2018-2026**, the model estimates win probabilities for all five badminton disciplines.

**LightGBM** combines **score-sensitive Elo** and **player-based TrueSkill** with:

- **Player strength:** individual ratings, doubles partnerships, and past match scores.
- **Match history:** recent form, head-to-head results, and rest days.
- **Tournament context:** event type, tournament level, round, and seeding.

Historical evaluation follows chronological order, with result-availability checks designed to reduce data leakage.
