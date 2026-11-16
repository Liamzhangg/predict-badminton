# Country Candidate vs Current Production

Main comparison: current production's 44-feature recipe retrained at matched historical cutoffs, versus the unchanged 55-feature country candidate. Five seeds; quarterly retraining is primary. Scores are per-seed means, not an ensemble. The actual saved artifact is never scored on its old training matches; its separate post-snapshot check is below. No production change.

| Evaluation | Bundle | Accuracy | Log Loss | Brier |
| --- | --- | ---: | ---: | ---: |
| validation | production_recipe | 76.206% | 0.478940 | 0.158751 |
| validation | country_candidate | 77.936% | 0.450125 | 0.148249 |
| annual_2026 | production_recipe | 76.150% | 0.480512 | 0.159423 |
| annual_2026 | country_candidate | 77.668% | 0.456492 | 0.150624 |
| rolling_2026 | production_recipe | 76.234% | 0.480095 | 0.159279 |
| rolling_2026 | country_candidate | 77.741% | 0.454972 | 0.150131 |

## Paired Changes

Candidate minus production recipe, crossed seed/tournament 95% intervals.

| Evaluation | Accuracy Change (pp) / 95% Interval | Log-Loss Change / 95% Interval |
| --- | --- | --- |
| validation | +1.730 / [+1.398, +2.083] | -0.028815 / [-0.033578, -0.024516] |
| annual_2026 | +1.518 / [+1.080, +1.978] | -0.024020 / [-0.029663, -0.018795] |
| rolling_2026 | +1.508 / [+1.063, +1.982] | -0.025123 / [-0.030761, -0.019892] |

## Actual Saved Artifact

Status: `verified_retrospective_comparison`. Restored the original 302,429-row training roster using historical observation timing. The reconstructed production booster is identical to the saved artifact, with maximum probability difference 0.0. The initial saved-receipt roster alone admitted 216 later-fetched older matches; that failed reconstruction is retained in artifact-recheck.json.

Only the 249 targets strictly after the 2026-11-12T05:42:12 production snapshot are evaluated. Both models use the same original training snapshot, 500 trees, parameters and seed. Features are refreshed before each target. This is a small already-seen subset of the historical benchmark, not prospective evidence or a rescore of the original saved betting forecasts.

| Actual-Artifact Diagnostic | Accuracy | Log Loss | Brier |
| --- | ---: | ---: | ---: |
| Saved production model | 74.699% | 0.483388 | 0.161053 |
| Country candidate, same training snapshot | 74.699% | 0.477536 | 0.158790 |

Accuracy ties at 186 correct predictions each. The loss/Brier point estimates favor the candidate, but only three tournament clusters are represented and the paired intervals include zero.

## Gates

- validation_accuracy_not_worse: PASS
- validation_log_loss_not_worse: PASS
- annual_2026_accuracy_not_worse: PASS
- rolling_accuracy_crossed_ci_strictly_positive: PASS
- rolling_log_loss_crossed_ci_not_worse: PASS
- rolling_brier_mean_not_worse: PASS
- Historical gate passed: True
- Prospective paired forward gate: NOT AVAILABLE. Candidate forecasts were not frozen before results.
- Production/default bundle and saved predictions/bets: unchanged. No candidate model exported or promoted.

## Decision

The country candidate passes every predeclared historical gate against the current production recipe. The primary quarterly replay improves accuracy by 1.508 percentage points (crossed 95% interval +1.063 to +1.982), log loss by 0.025123 and Brier by 0.009148. All five seeds improve accuracy. This is a strong historical candidate, not proof of future profit or complete production readiness.

The exact saved-artifact diagnostic is smaller and inconclusive on accuracy, and no pre-result country-candidate forecasts are available for genuine forward validation. Neither the active bundle nor saved forecasts/bets were changed. Calibration was not refit; descriptive ECE is mixed despite better log loss/Brier.
