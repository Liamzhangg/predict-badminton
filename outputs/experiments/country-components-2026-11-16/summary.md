# Country Elo, Identity and History Counts

Same frozen cohort, three seeds and observed-v2 timing as the earlier country-control study. Home advantage and distance are off throughout; location-coverage flags are held constant. Counts are separate from Elo ratings. Country categories are training-only. The reused 2026 benchmark is retrospective, not independent confirmation. No production changes.

Mean per-seed metrics, not ensemble scores. Lower log loss is better.

| Setting | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 76.282% | 0.478496 | 76.182% | 0.480284 |
| counts_only | 76.891% | 0.470019 | 76.734% | 0.471986 |
| elo_only | 77.460% | 0.458511 | 76.976% | 0.464784 |
| identity_only | 77.854% | 0.451009 | 77.638% | 0.457290 |
| elo_identity | 77.941% | 0.450793 | 77.618% | 0.456986 |
| original_combined | 77.922% | 0.450114 | 77.667% | 0.456250 |

Protocol, exact features, fold/seed scores and paired tournament-cluster intervals are in protocol.json/report.json. Baseline and original-combined anchors reproduced exactly in every seed/year. No models, feature matrices or per-match prediction arrays were retained.

## Interpretation

Country identity is the strongest individual component. Both Elo and identity help against baseline, but identity adds much more after Elo than the reverse. Adding identity to Elo reduces mean loss by 0.007719 in validation and 0.007798 in 2026, with all per-seed paired loss intervals below zero.

Adding Elo to identity reduces mean loss by only 0.000216 in validation and 0.000303 in 2026. Every validation interval includes zero; only one 2026 seed excludes zero, and the other two seeds are effectively neutral/slightly worse. Mean 2026 accuracy is 0.020 percentage points lower with Elo added.

Counts alone also help. Adding counts to both components improves mean loss by 0.000678 in validation and 0.000736 in 2026, but worsens 2026 loss in one seed. The original combined group still has the lowest average loss. Individual gains are not additive, and the marginal gains are much smaller than identity's standalone improvement.

This supports identity as the clearest individual signal, not automatic production promotion or a betting-profit claim. The benchmark has been reused; no independent prospective confirmation was obtained. Production, saved forecasts and existing bets were not changed.
