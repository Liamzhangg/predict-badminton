# Travel Distance Repeat Check

Three seeds (42, 7, 2026), identical frozen screening cohort and observed-v2 timing. Threshold selection uses pooled 2024-2025 results. The reused 2026 benchmark is descriptive, not new independent evidence. No production changes or model files saved.

Country-level great-circle estimates use the previous observed completed tournament until the cutoff, then the federation-home proxy. These are not actual player journeys. Inbound distance persists across rounds; missing locations stay missing.

Each score is the mean of three separately scored seeds, not an ensemble score. Lower log loss is better.

| Setting | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss | Loss Change vs Control |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 76.212% | 0.478904 | 76.146% | 0.480564 | +0.000000 |
| geography_14d | 76.397% | 0.476929 | 76.388% | 0.480072 | -0.000492 |
| home_only | 76.535% | 0.473072 | 76.634% | 0.474613 | -0.005952 |
| travel_7d | 76.419% | 0.475153 | 76.538% | 0.476613 | -0.003951 |
| travel_14d | 76.435% | 0.475190 | 76.481% | 0.476992 | -0.003573 |
| travel_21d | 76.455% | 0.475647 | 76.479% | 0.477495 | -0.003069 |
| travel_28d | 76.439% | 0.475471 | 76.531% | 0.477535 | -0.003029 |

Validation winner: `home_only`. Best cutoff on validation: `travel_7d`.

Per-seed paired differences and tournament-year bootstrap intervals are in report.json. The original seed-42 scores were reproduced exactly. All pre-existing files passed SHA-256 checks.
