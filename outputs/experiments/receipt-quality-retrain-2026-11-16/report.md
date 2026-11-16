# Receipt and Training Quality Refit

The receipt-handling bug is fixed and seven contradictory results are quarantined. Following explicit approval after the comparison, the cleaned model is now the production default for new forecasts. Historical predictive performance is essentially unchanged; this is a data-quality promotion, not a demonstrated accuracy gain. All Hong Kong forecasts and paper bets remain frozen.

## Changes

1. New result receipts require explicit `matchStatus=F`, which is now part of the result fingerprint. Live, scheduled, absent and unknown statuses cannot create completed-result receipts. Explicit live/scheduled reopening breaks an earlier completion episode. Valid played retirements remain supported.
2. Legacy receipts fail closed unless exact saved response bytes match the original response SHA-256 and prove the corresponding normalized result was final. All 34 distinct source-response hashes were found. Completion proofs were appended to 2,284 original receipt associations covering 609 matches, preserving every original availability time and database row. The pre-migration database is backed up locally.
3. The seven normal-status winner/score contradictions are excluded by match ID and canonical fixture signature. Duplicate aliases cannot restore them. Raw source records are unchanged, and no winner was guessed or flipped. Exclusions were based on source inconsistency, not whether a model predicted the match correctly.

The remaining 27 extra-game anomalies, one incomplete normal-status sequence, and broader missing/nonstandard score categories from the audit are not automatically declared wrong labels. They remain distinct from the seven confirmed internal contradictions and need source/format review.

Implementation: [receipt recording and verification](../../../predict_badminton/data_sources/observations.py), [parser exclusions](../../../predict_badminton/data_parser.py), [quarantine catalog](../../../data/quarantine/bwf_matches/records.csv), [regression tests](../../../tests/test_result_quality.py).

## Matched Historical Comparison

Primary evaluation is a 2026 quarterly replay with five fixed seeds: 42, 7, 2026, 13 and 99. Both variants use the active 55-feature country recipe, identical LightGBM hyperparameters, availability-purged chronological early stopping, training-only country categories and raw two-order-averaged probabilities. No new feature or hyperparameter search was performed.

In the tables, "Current Recipe" names the pre-cleanup production control, not the subsequently promoted default.

The exact pre-fix 302,941-row cohort and causal features were frozen before editing the parser. The cleaned cohort contains 302,934 rows, exactly seven fewer. Every retained label, result-availability time, prediction time and score margin matches the reference. Receipt verification removed no observed matches or changed their times; the cohort difference is the quarantine.

Both variants are scored on the same 44,188 non-quarantined 2026 fixtures. One quarantined 2026 target is excluded from both evaluations, not just the candidate's. Five seeds are averaged; they are not an ensemble or five independent sets of matches. Current-recipe means are reconstructed historical fits, not the all-data production artifact scored on its own training rows.

| Quarterly Replay Population | Matches | Current Recipe Accuracy | Cleaned Recipe Accuracy | Current Log Loss | Cleaned Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| All eligible 2026 targets | 44,188 | 77.739% | 77.727% | 0.455025 | 0.455410 |
| Senior World Tour | 4,559 | 72.779% | 72.814% | 0.526686 | 0.526814 |
| World Tour R32 | 2,106 | 74.340% | 74.311% | 0.502224 | 0.503142 |
| Junior tagged | 18,023 | 78.047% | 78.011% | 0.455776 | 0.456358 |

Lower log loss and Brier score are better. Pooled Brier is 0.150150 for the current recipe versus 0.150279 cleaned. Descriptive pooled calibration error is slightly lower for the cleaned recipe, 0.005627 versus 0.005916, but that alone does not establish better betting probabilities.

Paired changes below are cleaned minus current. Intervals use 5,000 tournament-edition/year resamples crossed with seed resampling. The pooled comparison contains 213 tournament clusters.

| Pooled Metric | Mean Change | Crossed 95% Interval |
| --- | ---: | ---: |
| Accuracy, percentage points | -0.0118 | [-0.1086, +0.0755] |
| Log loss | +0.000384 | [-0.000027, +0.000829] |
| Brier score | +0.000129 | [-0.000025, +0.000292] |

All these intervals include zero. Two seeds improve pooled accuracy and three decline. Senior World Tour accuracy changes by +0.0351 percentage points, with an exploratory interval of [-0.2351, +0.2886]. Subgroup intervals are not multiplicity-adjusted.

The secondary annual replay, reusing each January fit, also shows no clear gain: pooled accuracy is 77.665% current versus 77.624% cleaned, and log loss is 0.456548 versus 0.456695. Full subgroup and per-seed results are in [comparison.json](comparison.json).

**Interpretation:** the defects were real data-integrity problems, but these seven records and the latent receipt path do not explain a large inflation in the current headline accuracy. Cleaning does not produce a demonstrated accuracy or probability-quality improvement in this replay. Data-integrity correctness is independent of whether a reused benchmark score rises.

The benchmark years have been reused during development. This comparison does not remove adaptive selection bias, certify old publication times, establish profitability, or prove superiority on future World Tour markets. Its target cohort also differs slightly from the earlier 44,179-match promotion replay; use the matched current control here rather than subtracting scores from different snapshots.

## Cleaned Bundle

Candidate directory: `outputs/runs/2026-11-16-receipt-quality-cleaned`.

The final candidate uses all 302,934 cleaned eligible matches, the same frozen data cutoff as production (`2026-11-16T18:17:18Z`), seed 42 and the original 500-tree parameters. Its latest included result receipt is still `2026-11-15T16:06:00.961766Z`; it retains the locally stored Kumamoto and Kaohsiung finals. This isolates data-quality changes from new data or a changed feature recipe.

The 55-feature model and saved state load successfully through the normal bundle loader. This all-data refit has not been evaluated on its own historical training rows; the table above describes separate historical recipe fits.

- Candidate model SHA-256: `4c75695a51ae3bb5aa5489d23b151b5a9759f4466857e77143048e9705e3138a`
- Candidate state SHA-256: `d4fe1505cd3379a0f5f94c7a0a3d367a3645335228cb181bf44ea1855711b9df`
- [Candidate metadata](../../runs/2026-11-16-receipt-quality-cleaned/metrics_report.json)

During the comparison stage, production weights, state, backups and the default pointer remained unchanged. The subsequent explicitly approved promotion changes only the selector to the verified cleaned bundle and preserves the previous country selector for rollback. Existing model binaries are not overwritten. The shared loader/refresh code enforces the corrected receipt and quarantine rules for new work. Actual post-refit predictive accuracy is not yet available. See the separate [promotion record](promotion.json) for activation and preservation checks; the earlier protocol and verification describe the pre-promotion comparison stage.

## Private Records

Original forecasts, odds, paper bets and accounting remain frozen locally. Individual forecasts and shadow comparisons are not published. The aggregate model comparison above does not rescore, reselect or settle those records.

## Verification

All 313 repository tests pass after promotion, including twelve receipt/quarantine regression tests and the new default-selector preservation test. The comparison-stage suite had 312 tests. Checks cover provisional-to-final receipt reuse, reopened completion episodes, unknown statuses, unverifiable legacy receipts, forged or conflicting cached evidence, preserved legacy availability times, retirements and duplicate quarantine aliases.

The comparison-stage [preservation check](verification.json) verifies 15,206 pre-existing data/output files. At that stage the only intentional changes among those files were the result-observation database and `data/README.md`; every original database row remained intact, database integrity and foreign keys passed, and production/Hong Kong artifacts stayed frozen. The later approved selector change is checked separately in the [promotion record](promotion.json). Source code, tests and the new quarantine catalog are separate implementation changes.

Evidence: [locked protocol](protocol.json), [cohort check](cohort-verification.json), [database migration](database-migration.json), [candidate verification](candidate-verification.json), [paired results](comparison.json).

## Reproduction

The original control was captured with `prepare_reference.py` at commit `6c8c208` before the fixes. Its large feature/match snapshots and database backup remain local. With those snapshots and local raw/model inputs available:

```sh
PYTHONPATH=. .venv/bin/python outputs/experiments/receipt-quality-retrain-2026-11-16/run.py
PYTHONPATH=. .venv/bin/python outputs/experiments/receipt-quality-retrain-2026-11-16/verify.py
.venv/bin/python -m unittest discover -s tests -q
```

`verify.py` records the pre-promotion comparison stage and asserts the original selector, so it must not be rerun against the subsequently promoted default. `verify_promotion.py` produced the separate activation record using a local pre-promotion checksum inventory. Published scripts require the excluded local inputs and are not a self-contained trained environment.

Per-fit prediction checkpoints allow an interrupted replay to resume. Completed reports and candidate bundles are write-protected against accidental overwrite; use a separately dated experiment for a new run.
