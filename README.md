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

The active model uses **302,332 played matches from 2018-2026**, through November 8, 2026, with 44 features and 500 LightGBM trees:

- **Score-sensitive Elo:** individual strength and doubles partnership history. Complete normal scores scale updates from 0.5x to 1.5x; missing, retired, or invalid scores retain the normal update.
- **Player-based TrueSkill:** skill and uncertainty per discipline, carried between doubles partners.
- **Recent history:** win rates, match counts, rest days, and head-to-head results.
- **Context:** discipline, tournament tier, round, seeds, and country overlap.

LightGBM learns how to combine these inputs. Training excludes para, AirBadminton, unplayed walkovers, and invalid team formats; standard junior events remain included. There is no automatic Elo decay or extra calendar-based TrueSkill uncertainty: that experiment was not adopted.

**History is availability-aware.** The saved active model uses `conservative_day_history_v1`. Future training defaults to `observed_results_with_conservative_history_v2`: exact observed results use receipt times, while untracked older history keeps conservative day boundaries. Only results eligible strictly before each prediction cutoff enter history. See [timing rules](data/README.md#training-rules).

New training also uses the saved [official tournament catalog](data/catalogs/bwf_tournaments.csv) for World Tour tiers, joined by edition ID/code rather than tournament name. Other categories retain the old heuristic. The catalog is frozen into each new run; existing model bundles keep their original tier behavior. See [category mapping](data/README.md#official-categories).

**Predictions average both opponent orders:** `P(A wins) = (model(A,B) + 1 - model(B,A)) / 2`. Reversing the opponents produces complementary probabilities through the shared [prediction function](predict_badminton/probabilities.py). Optional calibration acts on each direction before averaging.
