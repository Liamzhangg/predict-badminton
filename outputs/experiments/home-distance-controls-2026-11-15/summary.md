# Home Advantage, Country Strength and Home-Country Distance

Three seeds, identical frozen cohort and observed-v2 timing. All factorial models include the same location-coverage flags. Country strength includes point-in-time country Elo plus training-only country identities. Distance is a country-level geographic proxy, not actual travel. Reused 2026 benchmark; no production promotion.

Scores below are mean per-seed metrics, not ensemble scores. Lower log loss is better.

| Variant | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss |
| --- | ---: | ---: | ---: | ---: |
| raw_control | 76.212% | 0.478904 | 76.146% | 0.480564 |
| baseline | 76.282% | 0.478496 | 76.182% | 0.480284 |
| home | 76.393% | 0.476905 | 76.412% | 0.480147 |
| country_strength | 77.922% | 0.450114 | 77.667% | 0.456250 |
| home_country_strength | 77.982% | 0.449616 | 77.578% | 0.456214 |
| distance | 76.523% | 0.473534 | 76.653% | 0.474692 |
| home_distance | 76.565% | 0.473292 | 76.650% | 0.474725 |
| country_strength_distance | 77.986% | 0.449786 | 77.649% | 0.455397 |
| all | 77.980% | 0.450073 | 77.714% | 0.455600 |

## Distance Beyond Both Controls

| Seed | 2026 Loss Change | 95% Tournament-Cluster Interval |
| --- | ---: | --- |
| 42 | -0.000739 | [-0.001276, -0.000188] |
| 7 | -0.001177 | [-0.001837, -0.000553] |
| 2026 | +0.000073 | [-0.000548, +0.000710] |

Definitions, formulas, split policies, per-seed metrics and all paired contrasts are in protocol.json/report.json. All original controls reproduced exactly. All protected pre-existing files remained unchanged. No models, feature matrices or per-match prediction arrays retained.

## Interpretation

Country controls explain most of the improvement: 2026 accuracy +1.484 percentage points and loss -0.024033 against the matched baseline. Those controls combine past-only country Elo and training-only country identity; this study does not separate their individual contributions. Home advantage alone has a small mean loss gain, with all per-seed loss intervals including zero.

Distance beyond both controls is mixed: validation mean loss +0.000457 (worse), versus 2026 -0.000614 (better) and accuracy +0.136 percentage points. It improves 2026 loss in two seeds but is neutral/slightly worse in the third; validation improves in only one seed. The reused 2026 benchmark does not override that inconsistency. No production promotion or claim of real travel fatigue or betting profit follows from this study.
