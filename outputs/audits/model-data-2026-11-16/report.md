# Badminton Model and Data Audit

Use 77.74% as a reused historical benchmark, not as a fresh estimate of Hong Kong or betting accuracy. The active model is current through the locally stored Kumamoto and Kaohsiung finals, but its superiority on senior World Tour markets is not established.

Audit date: 2026-11-16. Scope: active 55-feature country bundle, its exact 302,941-row training cohort, parser and result receipts, causal rating replay, model selection, chronological evaluation, and frozen Hong Kong targets. This was a read-only production review with separate diagnostic fits. No production model, source result, forecast, odds, or bet was changed.

## Findings

### P1 Repeated benchmark reuse makes the headline accuracy optimistic as validation evidence

The research record explicitly says the 2024-2026 benchmarks were reused during development. The latest promotion comparison also calls 2026 a reused benchmark. Those outcomes have influenced feature and model selection, even though each individual fit respects chronological boundaries. This is adaptive test-set overfitting: repeated choices can fit the evaluation period without putting the target winner into an input column.

The 77.741% accuracy, 0.454972 log loss and 0.150131 Brier score come from a five-seed historical quarterly replay. They are not a test of the final all-data refit, not five independent sets of matches, and not prospective performance. The artifact metadata correctly labels this, including `prospective_forward_validation: false`. Repeated seeds and tournament bootstrap intervals measure some sampling variation; they do not remove the bias from choosing features after seeing these outcomes.

The size of this optimism cannot be measured from the same reused data. A new frozen forward evaluation is required. This audit's retrospective diagnostics do not restore a virgin holdout.

Sources: [research limitations](../../../EXPERIMENTS.md#L8), [locked comparison protocol](../../experiments/country-vs-production-promotion-2026-11-16/run.py#L168), [artifact evaluation label](../../runs/2026-11-16-country-context-181718Z/build_bundle.py#L103).

### P1 Pooled gains do not establish a better model for senior World Tour betting

The training set contains 115,599 junior-tagged matches, 38.16% of its rows, plus many other non-World-Tour events. That is not intrinsically bad training data. It does mean pooled accuracy answers a different question from accuracy on the markets being quoted for Hong Kong.

Reusing the original five-seed quarterly replay, the country model's accuracy advantage is only 0.242 percentage points on 4,554 World Tour matches from 27 tournament clusters, versus 1.653 points on the other 39,625 matches. World Tour log loss is slightly worse by 0.000132; lower log loss means better probabilities. The exploratory crossed seed/tournament 95% accuracy interval is -0.206 to +0.682 percentage points. It includes no improvement. These subgroup intervals are not adjusted for adaptive model or subgroup selection.

A separate seed-42 annual replay gives the following diagnostic results. Both recipes train only on earlier available results, choose tree count using an internal chronological tail, and average both opponent orders. The country replay uses 345 trees selected internally; the baseline uses 500. These are historical recipe comparisons, not scores for today's final 500-tree artifact.

| 2026 subset | Matches | Baseline accuracy | Country accuracy | Baseline log loss | Country log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| All eligible matches | 44,189 | 76.18% | 77.58% | 0.480641 | 0.457234 |
| Senior World Tour | 4,559 | 72.34% | 72.58% | 0.527353 | 0.528088 |
| World Tour R32 importance group | 2,106 | 74.50% | 73.98% | 0.498584 | 0.504666 |
| Junior tagged | 18,023 | 75.40% | 77.75% | 0.494064 | 0.458856 |

The R32 group uses the model's round-importance value 5, which is also its fallback for unknown round names. The single annual subgroup decline is a warning, not sufficient evidence to demote the model automatically. Neither pooled accuracy nor these exploratory subgroup scores establish profitable positive-EV selection. Probability calibration must be checked specifically in the quoted-market population and probability ranges used for selections.

Sources: [primary subgroup sums](primary-cluster-subgroups.json#L1), [exploratory intervals](primary-world-tour-intervals.json#L1), [annual diagnostics](retrospective-diagnostics.json#L1), [round mapping](../../../predict_badminton/config.py#L50).

### P2 Provisional live results can acquire a completed-result receipt too early

`_snapshot()` accepts a winner without requiring final match status. `matchStatus` is absent from the fingerprint fields. A controlled reproduction records a live-status payload with a winner at 10:05, changes only its status to final, and loads it at a 10:10 historical cutoff. The final-status source is accepted with the old 10:05 receipt even though its recorded start plus duration is 11:00. The parser checks receipt time against prediction time, but not this completion-state transition.

The current refresh and country refit exclude raw live-status records. That does not prevent reusing their earlier receipts after status changes. The existing provisional-winner test covers raw live exclusion, not live-to-final receipt reuse.

Observed incidence: zero receipt-before-recorded-end cases in the current eligible source rows with valid UTC start and duration. The reproduction establishes a latent leak path, not that this bug inflated the current score. The recorded end is itself source data rather than an independently verified clock, so it is an audit signal, not a perfect completion oracle.

Required fix: accept a result receipt only with a verified completion signal and include relevant completion state in the result version; add live-to-final transition tests. Retirements need an explicit valid completed-result path rather than an assumption that every result has two won games.

Sources: [receipt fields](../../../predict_badminton/data_sources/observations.py#L18), [receipt acceptance](../../../predict_badminton/data_sources/observations.py#L40), [parser timing check](../../../predict_badminton/data_parser.py#L208), [existing live test](../../../tests/test_forecast_history.py#L106), [controlled reproduction](provisional-receipt-probe.json#L1).

### P2 Historical data vintages and publication times are mostly assumed

Only 609 of 302,941 training rows, 0.20%, use a recorded observed-result receipt. The remaining 302,332 use conservative reconstructed availability times:

| Availability rule | Rows |
| --- | ---: |
| Unverified local date with two-day embargo | 167,270 |
| UTC day after recorded end | 134,278 |
| Uncertain UTC timing with two-day embargo | 784 |
| Observed completed result | 609 |

The fallback rules deliberately delay result use, and unzoned dates deliberately move the target prediction earlier. This is substantially safer than processing by match start alone. It still cannot prove when an old result or a later correction was actually published. Current snapshots can also contain later metadata corrections. A historical replay using these snapshots cannot be certified as a full point-in-time data-vintage replay.

This is a provenance limitation, not a finding that 99.8% of rows leaked. Existing receipts are ingestion observations; their absence does not imply results were unavailable to the public then. Preserve exact source versions prospectively and test sensitivity to stricter historical embargoes before making stronger historical claims.

Sources: [timing rules](../../../predict_badminton/temporal.py#L11), [measured cohort timing](data-summary.json#L1).

### P2 Contradictory winner and score records remain eligible for labels and ratings

The refined source check found seven distinct normal-status matches whose completed best-of-three score identifies the opposite winner from the stored `winner`. All seven are selected canonical training records, and all retain that stored winner. For example, match 1304511 has scores 21-19, 21-12 but `winner=2`; match 1304513 has 15-21, 11-21 but `winner=1`.

The score-margin parser safely returns missing for these contradictions. The training parser nevertheless retains the winner as the label, and the rating update still uses that winner with a neutral score multiplier. Thus inconsistent data enters labels and subsequent ratings. The audit cannot determine whether the score, winner, or final-status metadata is wrong without checking the source; automatically flipping these winners would be unsafe.

There are also 27 source rows with an extra game after 2-0 and one normal-status 1-1 sequence with a declared winner. Separately, 10,660 normal-status source rows have missing or nonstandard 21-point sequences, and 2,452 have non-normal score status. Those broader categories include legitimate formats and retirements and are not automatically errors. The earlier broad screen's 87 flags must not be read as 87 wrong winners; the refined report supersedes that interpretation.

The seven direct contradictions are a very small fraction of the cohort. They establish a validation gap, not an explanation for the several-point headline score. Quarantine demonstrably inconsistent normal-format records pending source verification; keep valid retirements and alternative formats distinct.

Sources: [winner retained as label](../../../predict_badminton/data_parser.py#L244), [unsafe score margin rejected](../../../predict_badminton/match_rules.py#L65), [rating update](../../../predict_badminton/features.py#L665), [refined examples and canonical IDs](refined-score-checks.json#L1).

## Checks That Passed

1. Reconstructed the exact active 302,941-row cohort and matched its saved SHA-256. No winner, own score margin, duration, result timestamp or match ID is selected directly among the 55 model input columns.
2. Result release is strict before prediction; equal-timestamp targets are previewed before their outcomes update Elo, TrueSkill, form, H2H or country context. Chronological holdouts, folds and internal early stopping purge training labels unavailable at the boundary. Existing own/future-outcome and tied-time regression tests pass.
3. The parser removes 652 duplicate-fixture copies and 208 source aliases. The audit found no remaining repeated knockout groups with identical edition, discipline, round and opponents. Team ties were excluded from that extra screen because repeated slots can be legitimate. No obvious player-ID collision was found by disjoint frequent names or repeated MS/WS participation; this heuristic is not a complete identity proof.
4. Country categories are fitted on training rows, shared between opponent sides and frozen for inference. Unseen identities become missing. The bundle requires matching feature/state policies, and predictions average both opponent orders. Old calibrators are not attached to the new schema.
5. All 296 repository tests pass. `git diff --check` is clean. The preservation check covers 15,243 pre-existing data, output and source files; production artifacts, source data, forecasts and bets remain unchanged.

Relevant code: [causal feature replay](../../../predict_badminton/features.py#L716), [availability-purged splitting](../../../predict_badminton/model.py#L64), [frozen country encoding](../../../predict_badminton/country.py#L73), [country temporal tests](../../../tests/test_country.py#L84).

## Overfitting Interpretation

In the isolated annual country replay, in-sample training accuracy is 77.70% and 2026 accuracy is 77.58%; training log loss is 0.450428 versus 0.457234 on 2026. This does not show a large classic train/test gap. Different years have different match difficulty and data coverage, so the small gap is not proof that overfitting is absent.

Fixed predictions scored against shuffled test labels give 50.14% accuracy and 0.50007 AUC. That is only a label-alignment sanity check. It does not by itself test or rule out feature leakage. The temporal code review, cohort reconstruction and causal regression tests provide the more relevant checks.

The stronger unresolved concern is adaptive selection on reused years, combined with deployment-population mismatch. Current evidence supports a useful historical pooled predictor, not a demonstrated 77.74% forward hit rate or a proven best betting model.

## Freshness and Retraining

The active bundle is `outputs/runs/2026-11-16-country-context-181718Z`, refit on 302,941 eligible results at a frozen 2026-11-16 18:17:18 UTC cutoff. The latest included receipt is 2026-11-15 16:06:00.961766 UTC. This audit did not retrain or replace it; the two diagnostic fits were separate in-memory historical models.

The reconstructed training cohort includes 171 eligible Kumamoto 2026 results and 237 eligible Kaohsiung 2026 results, including all five discipline finals from each event. No locally stored final-status, non-walkover winner record from those events is missing from the cohort. This verifies local ingestion and training inclusion, not completeness against a new independent official crawl.

The frozen Hong Kong forecast selects this country bundle, refreshes eligible history at its own cutoff, and has 79 target IDs. None appears among training source IDs. These saved targets are suitable for prospective tracking only while their original probabilities and selection rules remain frozen.

Evidence: [recent events and target overlap](recent-events-and-targets.json#L1). Individual forecast metadata remains local-only.

## Recommended Next Actions

1. Fix and regression-test the provisional-to-final receipt path before relying on a stronger live-availability claim.
2. Verify and quarantine the seven contradictory normal-format records without guessing corrected winners. Rebuild into a new version only after the source decisions are recorded.
3. Freeze both country and baseline shadow forecasts for future senior World Tour fixtures. Lock the comparison period, target population, model versions and promotion criteria before outcomes arrive.
4. Judge the forward comparison using log loss, Brier score, calibration and tournament-level uncertainty, alongside accuracy. Track all eligible forecasts as well as selected bets so selection does not distort model accuracy.
5. Keep the current historical score explicitly labeled reused, preserve original predictions and bet records, and retain the prior bundle as a comparison. No automatic production switch is justified by this audit alone.

## Reproduction

Run from the repository root with the original local raw data, result receipts and model bundles available:

```sh
PYTHONPATH=. .venv/bin/python outputs/audits/model-data-2026-11-16/audit.py
PYTHONPATH=. .venv/bin/python outputs/audits/model-data-2026-11-16/source_validation.py
PYTHONPATH=. .venv/bin/python outputs/audits/model-data-2026-11-16/primary_subgroups.py
.venv/bin/python -m unittest discover -s tests -q
```

The diagnostic scripts write only within this audit directory. The primary script checks the original protected-file snapshot and fails if protected files differ. A later legitimate production or data update will therefore require a separately dated audit rather than overwriting the original preservation baseline.

Publication note: raw sources, model binaries, private betting records, individual predictions and the full protected-file inventory remain local. The published preservation verification records the completed audit-time check; only aggregate diagnostics and source code are published.
