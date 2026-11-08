# Model Results

Updated November 8, 2026. Data cutoff: October 21, 2026. Includes the new November 8 Glicko-2 comparison and earlier saved runs.
Higher accuracy and lower log loss are better. Compare rows within the same section only. Main model: **LightGBM + Elo + Glicko-2, raw probabilities**.
Promoted November 8 after full retraining reproduced the combined experiment; 47 features, 496 trees, production refit on all 300,103 played matches. History still ends October 21, not November 8.

## 2026 Holdout

39,537 matches; classifiers trained on pre-2026 data. Results describe evaluation models before the all-data production refit. This holdout has been reused for experiments, so it is not a fresh untouched test.

| Run Date | Approach | Accuracy | Log Loss | Verdict |
| --- | --- | ---: | ---: | --- |
| 2026-10-21 | LightGBM + Elo, raw | 74.7958% | 0.501824 | Previous main model / control |
| 2026-10-21 | Control + Platt calibration | 74.7958% | 0.507817 | Worse probability quality |
| 2026-10-21 | Control + isotonic calibration | 74.7882% | 0.506375 | Worse probability quality |
| 2026-10-22 | Control + pair-only decay, 365-day half-life | 74.8388% | 0.502005 | No convincing gain; not adopted |
| 2026-11-08 | LightGBM + Glicko-2, no Elo rating values | 74.6314% | 0.499644 | 65 fewer correct; slightly better log loss |
| 2026-11-08 | LightGBM + Elo + Glicko-2, raw | 75.7898% | 0.485326 | Main model; retraining reproduces experiment |
| 2026-11-08 | Combined + Platt calibration | 75.7771% | 0.492305 | Worse than combined raw; optional |
| 2026-11-08 | Combined + isotonic calibration | 75.7417% | 0.488339 | Worse than combined raw; optional |

Pair-only decay: 17 more correct predictions, but worse log loss; the accuracy difference's approximate 95% interval includes zero (-0.111 to +0.183 percentage points).
Elo + Glicko-2: +0.994 percentage points in accuracy (393 more correct); approximate 95% tournament-bootstrap interval +0.687 to +1.319 points. The control rerun reproduced the saved control probabilities exactly. Promotion is based on the validation comparison plus this reused holdout; fresh later-data evaluation is still needed.

## Glicko-2 Validation

Run November 8, 2026. Same pooled 2024-2025 validation: 91,150 matches, each year trained only on earlier years. Same LightGBM defaults, seed, and chronological early stopping; no new calibration or parameter search.

| Approach | Accuracy | Log Loss | Verdict |
| --- | ---: | ---: | --- |
| LightGBM + Elo control | 74.8963% | 0.498493 | Control |
| LightGBM + Glicko-2 | 75.1421% | 0.495133 | Modest validation improvement |
| LightGBM + Elo + Glicko-2 | 76.0121% | 0.483535 | Best validation result; selected before the 2026 check |

Glicko-2 uses weekly updates and separate ratings for singles players or exact doubles partnerships in each event. New pairs start fresh; generic experience counts and non-rating context remain. These are LightGBM feature comparisons, not standalone Glicko win-probability tests.

## Decay Validation

Run October 22, 2026. Pooled 2024-2025 validation: 91,150 matches, each year trained only on earlier years. All approaches use LightGBM + Elo; decay starts after 90 inactive days.

| Approach | Accuracy | Log Loss | Verdict |
| --- | ---: | ---: | --- |
| No decay | 74.8963% | 0.498493 | Control |
| Player + pair decay, 730-day half-life | 74.6122% | 0.502637 | Worse |
| Player + pair decay, 365-day half-life | 74.4608% | 0.506095 | Worse |
| Player + pair decay, 180-day half-life | 73.9199% | 0.511567 | Worse |
| Pair-only decay, 365-day half-life | 74.8941% | 0.498344 | Near tie; selected for the 2026 comparison |

## 2025 Benchmarks

Run October 21, 2026. Same 51,323-match 2025 validation fold, trained on earlier years. These are not 2026 holdout results.

| Approach | Accuracy | Log Loss | Verdict |
| --- | ---: | ---: | --- |
| Elo-difference logistic regression | 68.6982% | 0.580748 | Simple baseline |
| LightGBM + full features | 74.8008% | 0.500669 | Better than Elo-only baseline |
| XGBoost + full features | 74.8047% | 0.499865 | Slightly better on this fold; not adopted |

## Not Tested Yet

- **TrueSkill or timezone/travel features:** discussed only; no measured results.

## Keeping This Updated

Append one row per completed run with its date, data cutoff, evaluation period/count, accuracy, log loss, and a short verdict. Preserve earlier rows; use a separate section when data or evaluation splits change.
Select changes using validation data, then check them on a reserved later period. Do not promote a model based only on a tiny accuracy increase or a cleaner-looking leaderboard.

Sources (local, ignored by Git): `outputs/runs/2026-10-21-french-open/metrics_report.json`, `outputs/experiments/elo-decay-2026-10-22/report.json`, `outputs/experiments/glicko2-2026-11-08/report.json`, and `outputs/runs/2026-11-08-elo-glicko2/metrics_report.json`.
