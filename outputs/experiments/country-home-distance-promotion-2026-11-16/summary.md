# Country vs Country + Home Advantage + Distance

Five fixed seeds, matched frozen cohort, raw two-order probabilities. Quarterly 2026 retraining is the primary replay; annual scores replicate the earlier experiment. All outcomes are reused historical results, not prospective forward validation. Nothing was promoted.

| Evaluation | Bundle | Accuracy | Log Loss | Brier | ECE (10 bins) |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | country_only | 77.936% | 0.450125 | 0.148249 | 0.006999 |
| validation | home_distance | 77.970% | 0.449990 | 0.148184 | 0.007849 |
| annual_2026 | country_only | 77.668% | 0.456492 | 0.150624 | 0.005720 |
| annual_2026 | home_distance | 77.683% | 0.455624 | 0.150376 | 0.005844 |
| rolling_2026 | country_only | 77.741% | 0.454972 | 0.150131 | 0.005908 |
| rolling_2026 | home_distance | 77.716% | 0.454685 | 0.150043 | 0.005801 |

## Paired Differences

Expanded minus country-only; accuracy is in percentage points. Crossed 95% intervals resample tournament editions and training seeds.

| Evaluation | Accuracy Change / 95% Interval | Log-Loss Change / 95% Interval |
| --- | --- | --- |
| validation | +0.034 / [-0.061, +0.130] | -0.000136 / [-0.000879, +0.000612] |
| annual_2026 | +0.015 / [-0.119, +0.146] | -0.000868 / [-0.001591, -0.000124] |
| rolling_2026 | -0.026 / [-0.168, +0.106] | -0.000287 / [-0.000865, +0.000294] |

## Fixed Gates

- validation_accuracy_not_worse: PASS
- validation_log_loss_not_worse: PASS
- annual_2026_accuracy_not_worse: PASS
- rolling_accuracy_crossed_ci_strictly_positive: NOT MET
- rolling_log_loss_crossed_ci_not_worse: NOT MET
- rolling_brier_mean_not_worse: PASS
- Prospective forward gate: NOT AVAILABLE. No verified pre-result records for these exact candidates.
- Historical gate passed: False
- Evidence ready for production promotion: False. Production was not changed.

No model bundles, feature matrices or per-match probability arrays retained. Compact per-cluster paired sums preserve the uncertainty calculation. Protocol, source audit and detailed scores are alongside this report.

## Decision

The expanded bundle does not pass the fixed promotion check. Its five-seed annual accuracy gain is only +0.015 percentage points, with an interval spanning zero. Under quarterly retraining, the mean accuracy change is -0.026 points, with a crossed 95% interval of [-0.168, +0.106]. This does not establish a reliable improvement or prove that the package is worse.

Quarterly log loss and Brier improve slightly on average, but the paired loss interval also includes possible worsening. The accuracy and log-loss confidence gates are therefore not met. No genuinely prospective candidate records exist in the repository, so that separate gate cannot pass either. Neither candidate was promoted; production and existing predictions/bets remain frozen.
