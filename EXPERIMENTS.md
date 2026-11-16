# Experiments

Compact research record: one row per tested setting. Accuracy is higher-is-better; log loss is lower-is-better and measures probability quality. Each score cell is **accuracy / log loss**. `--` means not reported, not zero.

## Read This First

- Compare candidates only with the control in the same section. Data scope, cutoffs, classifiers and timing rules changed between studies; the highest number in this file is not a current-model claim.
- All 2024-2026 benchmarks were reused during development. Older studies use start-order timing and sometimes OTHER events; they predate later leakage/deduplication fixes. Lower loss here is descriptive, not proof of significance, prospective performance or betting profit.
- Validation is pooled 2024-2025 unless stated otherwise; holdout is 2026 unless a narrower slice is named. Five-seed entries are means across seeds 7/21/42/123/2026, not the best seed and not an ensemble.
- 'Helped' compares loss to the section control (within 0.000001 is essentially unchanged), flags accuracy tradeoffs, and uses validation only when no holdout was scored. A validation winner is not automatically a production change.
- Evidence is preserved locally under `outputs/experiment_reports/<original-run>/` as original reports, settings, study scripts and small summaries. Those files are Git-ignored; this Markdown is the portable tracked summary. Archived study scripts may need their derived inputs rebuilt before reuse.

## Current Feature Baseline

New forecasts use LightGBM with score-sensitive player/global/hybrid-pair Elo, player-based TrueSkill, experience/reliability counts, 5/10-match win rates, rest days, H2H wins/count/win rate, tournament tier, round, seeds, same-country flag and discipline, plus country Elo, country identity, intercountry counts and three coverage indicators. Both opponent orders are averaged. H2H has no manual extra weight. No home advantage, travel distance, calendar decay, workload additions, Glicko or calendar-inactivity TrueSkill growth is active.

The selected forecast bundle is `outputs/runs/2026-11-16-receipt-quality-cleaned`; it refreshes observed eligible history separately and excludes the seven quarantined records. `outputs/forecast_bundle.previous.json` retains the previous country bundle for rollback. The older observed-timing bundle and `outputs/artifacts` are untouched. The promotions are documented below. The [model and data audit](outputs/audits/model-data-2026-11-16/report.md) explains why reused pooled scores do not establish superiority on future senior World Tour markets.

## Elo inactivity decay

90-day grace period; decay rating offsets toward their priors. Legacy mixed-event benchmark, including OTHER.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/elo-decay-2026-10-22/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `none`: No calendar decay | 74.896% / 0.498493 | 74.796% / 0.501824 | Control |
| `both_730`: Player + pair Elo: 730-day half-life | 74.612% / 0.502637 | -- | Validation only: Higher loss |
| `both_365`: Player + pair Elo: 365-day half-life | 74.461% / 0.506095 | -- | Validation only: Higher loss |
| `both_180`: Player + pair Elo: 180-day half-life | 73.920% / 0.511567 | -- | Validation only: Higher loss |
| `pairs_365`: Pair Elo only: 365-day half-life | 74.894% / 0.498344 | 74.839% / 0.502005 | Higher loss despite higher accuracy; selected on validation |

## Elo versus Glicko-2

Glicko: exact teams by discipline, weekly batches, RD 350, volatility 0.06, tau 0.5. Legacy mixed-event benchmark; player-based TrueSkill later replaced Glicko.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/glicko2-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `elo_control`: Elo + existing context | 74.896% / 0.498493 | 74.796% / 0.501824 | Control |
| `glicko2`: Replace Elo values with Glicko-2 | 75.142% / 0.495133 | 74.631% / 0.499644 | Lower loss; accuracy lower |
| `elo_glicko2`: Elo + Glicko-2 | 76.012% / 0.483535 | 75.790% / 0.485326 | Lower loss; selected on validation |

## Glicko rating periods

Same Elo + Glicko model; only the period length changes. Legacy mixed-event benchmark.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/glicko-periods-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `7`: 7-day periods | 76.012% / 0.483535 | 75.790% / 0.485326 | Control |
| `1`: 1-day periods | 75.922% / 0.482603 | 75.764% / 0.486117 | Higher loss; selected on validation |
| `14`: 14-day periods | 75.806% / 0.485588 | 75.610% / 0.487196 | Higher loss |

## H2H and feature-group weighting

Split weights change training split preference, not a fixed share of the final probability. H2H adjustment: sigmoid(logit(p) + beta*n/(n+5)*ln((wins1+2)/(wins2+2))). Other groups stay at 1x. Legacy Elo + Glicko benchmark; manual weights are not used by the current model.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/feature-weighting-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: All groups 1x | 76.012% / 0.483535 | 75.790% / 0.485326 | Control |
| `h2h_0x`: h2h split-gain weight 0x | 75.906% / 0.484384 | -- | Validation only: Higher loss |
| `h2h_0.5x`: h2h split-gain weight 0.5x | 76.042% / 0.483251 | 75.641% / 0.485350 | Higher loss; selected on validation |
| `h2h_2x`: h2h split-gain weight 2x | 75.953% / 0.483695 | 75.737% / 0.486171 | Higher loss |
| `h2h_4x`: h2h split-gain weight 4x | 75.900% / 0.484303 | -- | Validation only: Higher loss |
| `elo_0.75x`: elo split-gain weight 0.75x | 75.866% / 0.483822 | -- | Validation only: Higher loss |
| `elo_1.25x`: elo split-gain weight 1.25x | 75.925% / 0.483619 | -- | Validation only: Higher loss |
| `glicko_0.75x`: glicko split-gain weight 0.75x | 75.979% / 0.484043 | -- | Validation only: Higher loss |
| `glicko_1.25x`: glicko split-gain weight 1.25x | 76.013% / 0.484073 | -- | Validation only: Higher loss despite higher accuracy |
| `form_rest_0.5x`: form_rest split-gain weight 0.5x | 75.899% / 0.484171 | -- | Validation only: Higher loss |
| `form_rest_2x`: form_rest split-gain weight 2x | 75.933% / 0.485153 | -- | Validation only: Higher loss |
| `h2h_logodds_0.1`: Extra H2H log-odds adjustment beta=0.1 | 76.016% / 0.483545 | 75.800% / 0.485389 | Higher loss despite higher accuracy |
| `h2h_logodds_0.25`: Extra H2H log-odds adjustment beta=0.25 | 76.009% / 0.483576 | -- | Validation only: Higher loss |
| `h2h_logodds_0.5`: Extra H2H log-odds adjustment beta=0.5 | 76.003% / 0.483664 | -- | Validation only: Higher loss |

## Workload and recovery

Global per-player workload; use team maxima/minimum recovery and only completed durations. Competition span uses <=7-day gaps, not actual travel. Legacy Elo + Glicko benchmark.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/workload-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: Existing ratings/form/context | 76.012% / 0.483535 | 75.790% / 0.485326 | Control |
| `match_counts`: Add matches in past 7/14 days | 75.965% / 0.483008 | 75.823% / 0.484754 | Lower loss |
| `full_workload`: Add 48h/7d/14d matches, 7d/14d minutes, recovery hours, active days, 3-game matches, other tournaments in 28d, competition span and coverage flags | 76.210% / 0.481587 | 75.972% / 0.483167 | Lower loss; selected on validation |

## Recent form and location history

Recent WR=(weighted wins+2)/(weighted matches+4), plus Elo-adjusted surprise/support. Location: other editions only, 180-day half-life, drop editions >730 days old, WR/surprise/support/titles/SF appearances. Unknown hosts remain missing; legacy benchmark, not current timing rules.

Reported sample counts: validation 91,150; holdout 39,537.
Local evidence: `outputs/experiment_reports/recent-location-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: Existing ratings/form/context | 76.012% / 0.483535 | 75.790% / 0.485326 | Control |
| `recent_7d`: Recent form: 7-day half-life | 75.982% / 0.483001 | -- | Validation only: Lower loss; accuracy lower |
| `recent_30d`: Recent form: 30-day half-life | 75.991% / 0.482716 | 75.899% / 0.484798 | Lower loss |
| `country_history`: Host-country history | 76.046% / 0.482766 | 75.557% / 0.486189 | Higher loss |
| `event_history`: Annual-event history | 75.941% / 0.483662 | -- | Validation only: Higher loss |
| `country_and_event`: Country + annual-event history | 75.985% / 0.483452 | -- | Validation only: Lower loss; accuracy lower |
| `recent_7d_and_locations`: 7-day form + both location histories | 75.988% / 0.482791 | -- | Validation only: Lower loss; accuracy lower |
| `recent_30d_and_locations`: 30-day form + both location histories | 76.004% / 0.482082 | 75.782% / 0.485319 | Lower loss; accuracy lower; selected on validation |

## One shared model versus five event models

Same Elo + Glicko features; five classifiers for MS/WS/MD/WD/XD versus one pooled classifier. Evaluation excludes OTHER on both sides; shared training still includes it. Current model remains shared.

Reported sample counts: validation 90,836; holdout 39,396.
Local evidence: `outputs/experiment_reports/discipline-models-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `shared`: One shared classifier | 76.017% / 0.483486 | 75.797% / 0.484942 | Control; selected on validation |
| `separate`: Separate MS/WS/MD/WD/XD classifiers | 75.724% / 0.488464 | 75.645% / 0.490702 | Higher loss |

## Additional-results replay: fresh_requested

232 matches; frozen shared classifier versus workload classifier, with history advanced between matches. Retrospective replay, NOT archived pre-match predictions or proof of a durable gain. Cohort names distinguish requested/newly collected versus reused 2026 fixtures.

Reported sample counts: validation not recorded here; holdout 232.
Local evidence: `outputs/experiment_reports/recent-results-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current_frozen`: Frozen shared classifier | -- | 74.569% / 0.549158 | Control |
| `workload_frozen`: Frozen workload classifier | -- | 75.000% / 0.543430 | Lower loss |

## Additional-results replay: fresh_all_collected

4,017 matches; frozen shared classifier versus workload classifier, with history advanced between matches. Retrospective replay, NOT archived pre-match predictions or proof of a durable gain. Cohort names distinguish requested/newly collected versus reused 2026 fixtures.

Reported sample counts: validation not recorded here; holdout 4,017.
Local evidence: `outputs/experiment_reports/recent-results-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current_frozen`: Frozen shared classifier | -- | 76.400% / 0.488093 | Control |
| `workload_frozen`: Frozen workload classifier | -- | 76.649% / 0.487374 | Lower loss |

## Additional-results replay: requested_retrospective

460 matches; frozen shared classifier versus workload classifier, with history advanced between matches. Retrospective replay, NOT archived pre-match predictions or proof of a durable gain. Cohort names distinguish requested/newly collected versus reused 2026 fixtures.

Reported sample counts: validation not recorded here; holdout 460.
Local evidence: `outputs/experiment_reports/recent-results-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `shared_pre2026`: Frozen shared classifier | -- | 74.130% / 0.525742 | Control |
| `workload_pre2026`: Frozen workload classifier | -- | 75.652% / 0.522747 | Lower loss |

## Additional-results replay: extended_2026_retrospective

43,861 matches; frozen shared classifier versus workload classifier, with history advanced between matches. Retrospective replay, NOT archived pre-match predictions or proof of a durable gain. Cohort names distinguish requested/newly collected versus reused 2026 fixtures.

Reported sample counts: validation not recorded here; holdout 43,861.
Local evidence: `outputs/experiment_reports/recent-results-2026-11-08/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `shared_pre2026`: Frozen shared classifier | -- | 75.785% / 0.485871 | Control |
| `workload_pre2026`: Frozen workload classifier | -- | 75.976% / 0.484034 | Lower loss |

## Score-sensitive Elo update rules

m=(winner points-loser points)/total points; K *= clip(1 + beta*(m-0.20)/0.20, 1-beta, 1+beta). Constant controls use pre-2024 K-weighted means; invalid scores use 1x. Original mixed-event benchmark.

Reported sample counts: validation 91,150; holdout 43,861.
Local evidence: `outputs/experiment_reports/score-margin-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: No score multiplier | 76.012% / 0.483535 | 75.785% / 0.485871 | Control |
| `point_mild`: Point margin beta=0.25 | 76.038% / 0.482143 | -- | Validation only: Lower loss |
| `mean_k_point_mild`: Constant-K control: 0.9863973249x | 76.015% / 0.483993 | -- | Validation only: Higher loss despite higher accuracy |
| `point_strong`: Point margin beta=0.50 | 76.128% / 0.480505 | 75.892% / 0.484287 | Lower loss; selected on validation |
| `mean_k_point_strong`: Constant-K control: 0.9727946499x | 75.990% / 0.483616 | 75.758% / 0.486540 | Higher loss |
| `games_only`: 2-0 => 1.15x K; 2-1 => 0.85x K | 75.950% / 0.483500 | -- | Validation only: Lower loss; accuracy lower |
| `mean_k_games_only`: Constant-K control: 1.0735573434x | 75.993% / 0.483690 | -- | Validation only: Higher loss |

## Score-sensitive Elo: five-seed confirmation

Five-seed means (7, 21, 42, 123, 2026), not an ensemble. Original mixed-event scope; point_strong is the rule later adopted, not a guarantee that its old accuracy transfers to the corrected pipeline.

Reported sample counts: validation 91,150; holdout 43,861.
Local evidence: `outputs/experiment_reports/score-margin-seeds-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: No score multiplier | 75.963% / 0.483736 | 75.730% / 0.486208 | Control |
| `point_strong`: Point-margin beta=0.50 | 76.128% / 0.480550 | 75.901% / 0.484178 | Lower loss |

## Score-sensitive Elo: late frozen slice

Five-seed means on the saved late-frozen slice, not a full-year 2026 score.

Reported sample counts: validation not recorded here; holdout 4,017.
Local evidence: `outputs/experiment_reports/score-margin-seeds-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: control | -- | 76.415% / 0.487631 | Control |
| `point_strong`: point strong | -- | 76.953% / 0.483867 | Lower loss |

## Score-sensitive Elo: standard-event scope check

Exclude unsupported event formats; seed 42, MS/WS/MD/WD/XD only. Same-point-margin implementation checked before adoption.

Reported sample counts: validation 90,621; holdout 43,580.
Local evidence: `outputs/experiment_reports/score-sensitive-promotion-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: No score multiplier | 75.946% / 0.483651 | 75.702% / 0.485506 | Control |
| `score_sensitive`: Point-margin beta=0.50 | 76.176% / 0.479789 | 75.909% / 0.483699 | Lower loss |

## TrueSkill strength estimates

Five-seed means (7, 21, 42, 123, 2026), not an ensemble. Standard events, legacy start-order timing. mu=25, sigma=25/3, beta=25/6, tau=25/300, draw probability=0; pair model starts each new partnership fresh.

Reported sample counts: validation 90,621; holdout 43,580.
Local evidence: `outputs/experiment_reports/trueskill-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current`: Then-current Elo + Glicko control | 76.139% / 0.480242 | 75.905% / 0.483737 | Control |
| `trueskill_pairs`: Elo + exact-pair TrueSkill | 76.166% / 0.478388 | 76.037% / 0.481862 | Lower loss |
| `trueskill_players`: Elo + player-based TrueSkill (adopted) | 76.673% / 0.470949 | 76.527% / 0.475165 | Lower loss; selected on validation |

## TrueSkill calendar inactivity

Five-seed means (7, 21, 42, 123, 2026), not an ensemble. Per-player/event sigma'=sqrt(sigma^2 + annual_sigma^2*max(gap-grace,0)/365.25), capped at initial sigma without reducing an already-higher sigma; means unchanged by the gap. Standard tau remains. No calendar-growth rule was adopted.

Reported sample counts: validation 90,621; holdout 43,580.
Local evidence: `outputs/experiment_reports/trueskill-inactivity-2026-11-09/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current`: No calendar growth (retained) | 76.673% / 0.470949 | 76.527% / 0.475165 | Control |
| `mild_30d`: Annual sigma=1; grace=30 days | 76.712% / 0.469764 | 76.631% / 0.473619 | Lower loss |
| `moderate_30d`: Annual sigma=2; grace=30 days | 76.679% / 0.470002 | 76.575% / 0.473763 | Lower loss |
| `strong_30d`: Annual sigma=4; grace=30 days | 76.546% / 0.471800 | 76.501% / 0.474936 | Lower loss; accuracy lower |
| `moderate_14d`: Annual sigma=2; grace=14 days | 76.690% / 0.469681 | 76.641% / 0.473536 | Lower loss; selected on validation |

## Opponent-order symmetry

Five-seed means, raw probabilities, same 43,580 2026 fixtures. combined=(p(A,B)+1-p(B,A))/2, adopted later. Pooled validation is not reported in this study; full 2024/2025 results remain in the evidence JSON.

Reported sample counts: validation not recorded here; holdout 43,580.
Local evidence: `outputs/experiment_reports/order-symmetry-2026-11-10/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `original`: Original order only | -- | 76.527% / 0.475165 | Control |
| `reverse_only`: Reverse order only | -- | 76.369% / 0.476766 | Higher loss |
| `combined`: Average both opponent orders (adopted) | -- | 76.551% / 0.473591 | Lower loss |

## Historical Calibration Checks

Seed-42 historical promotion check, 43,580 2026 fixtures. These are saved calibration scores from that promotion, not independent new experiments. Raw remains the forecasting default; current serving calibrates each direction before averaging when calibration is explicitly requested.

Reported sample counts: validation not recorded here; holdout 43,580.
Local evidence: `outputs/experiment_reports/order-symmetry-promotion-2026-11-10/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `raw`: raw | -- | 76.549% / 0.473802 | Control |
| `platt`: platt | -- | 76.549% / 0.480877 | Higher loss |
| `isotonic`: isotonic | -- | 76.572% / 0.474689 | Higher loss despite higher accuracy |

## Causal result-availability and duplicate corrections

Corrected conservative availability, deduplication and boundary purging; 15 holdout rows purged. This retained candidate score has NO matched pre-fix control; it cannot establish an accuracy gain from the safety corrections.

Reported sample counts: validation not recorded here; holdout 43,580.
Local evidence: `outputs/experiment_reports/availability-promotion-2026-11-11/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `corrected`: Corrected conservative pipeline | -- | 76.221% / 0.480229 | Safety correction; no matched control |

## Observed-result timing v2

Same 43,675 fixtures; raw two-order probabilities, seed 42. Loss delta +0.0000004134, accuracy unchanged: a safety/history policy, not a demonstrated accuracy improvement. This bundle is NOW selected for new forecasts; the old production artifacts remain unchanged.

Reported sample counts: validation not recorded here; holdout 43,675.
Local evidence: `outputs/experiment_reports/timing-v2-2026-11-12/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current`: Conservative history control | -- | 76.213% / 0.480338 | Control |
| `candidate`: Observed receipts + conservative fallback (selected) | -- | 76.213% / 0.480338 | Essentially unchanged loss |

## Official tournament tiers: all_standard_matches

Same timing, parameters and seed 42 within this slice. Registry maps World Tour Finals/1000/750/500/300 to tiers 2/2/3/4/5; other events retain heuristics. Supported by new training, but this candidate is NOT the currently selected forecast bundle.

Reported sample counts: validation not recorded here; holdout 43,677.
Local evidence: `outputs/experiment_reports/world-tour-categories-2026-11-12/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current_name_tiers`: Name-based tiers | -- | 76.214% / 0.480344 | Control |
| `official_categories`: Official edition categories | -- | 76.148% / 0.480208 | Lower loss; accuracy lower |

## Official tournament tiers: world_tour_all_rounds

Same timing, parameters and seed 42 within this slice. Registry maps World Tour Finals/1000/750/500/300 to tiers 2/2/3/4/5; other events retain heuristics. Supported by new training, but this candidate is NOT the currently selected forecast bundle.

Reported sample counts: validation not recorded here; holdout 4,485.
Local evidence: `outputs/experiment_reports/world-tour-categories-2026-11-12/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current_name_tiers`: Name-based tiers | -- | 72.441% / 0.526126 | Control |
| `official_categories`: Official edition categories | -- | 72.486% / 0.525250 | Lower loss |

## Official tournament tiers: world_tour_main_draw

Same timing, parameters and seed 42 within this slice. Registry maps World Tour Finals/1000/750/500/300 to tiers 2/2/3/4/5; other events retain heuristics. Supported by new training, but this candidate is NOT the currently selected forecast bundle.

Reported sample counts: validation not recorded here; holdout 4,032.
Local evidence: `outputs/experiment_reports/world-tour-categories-2026-11-12/`.

| Setting / Feature | Validation Accuracy / Loss | Holdout Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `current_name_tiers`: Name-based tiers | -- | 72.421% / 0.525634 | Control |
| `official_categories`: Official edition categories | -- | 72.619% / 0.524781 | Lower loss |

## Other Preserved Checks

Glicko/TrueSkill promotion checks verified that native code reproduced the corresponding experiments; they are not additional independent feature gains. First-seen receipt instrumentation is a timing safeguard, not an accuracy-tested feature. Their compact verification records remain with the evidence.

Operational odds transcription, R32 recovery, round forecasting and bet-settlement folders are not feature experiments. They remain at their original paths so saved forecasts, ledger source links, odds, probabilities and results are unchanged. No new predictions or settlements were run for this summary.

## Retention Rule

For a new study, add the exact setting/formula, matched control/candidate accuracy and loss, sample/split policy, seed aggregation and a short conclusion here. Preserve its small metrics/configuration report. Remove derived feature matrices, probability arrays, redundant model copies and logs once the study is recorded; never remove a selected bundle or a saved betting record as experiment clutter.

## Travel Distance (Country Proxy)

Separate observed-v2 timing study, seed 42; same matches and fixed LightGBM settings. Validation is pooled 2024-2025; the fallback was locked before scoring the reused 2026 benchmark. No production bundle, default pointer or saved bets changed.

Estimate great-circle km from the last known completed tournament's country, falling back to the current federation/home-country proxy after >7/14/21/28 days without play or with no prior history. A home-only control ignores the prior tournament. Preserve inbound distance across rounds of a continuous tournament; doubles average partner estimates. Only earlier available results update previous locations. Unknown countries remain missing.

These are country representative points, not actual flights, residences, cities or venue coordinates. England/Scotland/Wales use GB. Host-country metadata covers 97.71%; both 14-day distances are known for 93.65% of all study matches. Geography-only flags control for home status, origin fallback and missingness.

Coordinates: [Google country reference](https://developers.google.com/public-data/docs/canonical/countries_csv); country-code reference: [GeoNames](https://download.geonames.org/export/dump/countryInfo.txt). Local protocol, original sources, scripts, tests, matched predictions and detailed metrics: `outputs/experiments/travel-distance-2026-11-15/`.

Matched counts: validation 90,167; 2026 holdout 44,179. Score cells are accuracy / log loss. Unselected holdout variants were not scored.

| Setting | Validation Accuracy / Loss | 2026 Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| `control`: Existing 44-feature control | 76.191% / 0.478808 | 76.188% / 0.480568 | Control |
| `geography_14d`: Home/origin/missingness flags only | 76.372% / 0.477148 | 76.353% / 0.480360 | Lower loss |
| `home_only`: Always start from federation-home country | 76.554% / 0.473310 | 76.670% / 0.474646 | Lower loss; selected on validation |
| `travel_7d`: Fallback after >7 days | 76.435% / 0.475108 | -- | Validation only: Lower loss |
| `travel_14d`: Fallback after >14 days | 76.439% / 0.475331 | 76.462% / 0.476956 | Lower loss |
| `travel_21d`: Fallback after >21 days | 76.447% / 0.475602 | -- | Validation only: Lower loss |
| `travel_28d`: Fallback after >28 days | 76.383% / 0.475610 | -- | Validation only: Lower loss |
| `travel_14d_monotone`: 14 days; force own longer distance to lower win chance | 76.394% / 0.477285 | -- | Validation only: Lower loss |

Paired 2026 differences versus the control (candidate minus control; 2,000 tournament-year bootstrap resamples):

| Candidate | Accuracy Change (pp) | Log-Loss Change | 95% Cluster Interval, Loss |
| --- | ---: | ---: | --- |
| `geography_14d` | +0.165 | -0.000208 | [-0.001585, +0.001277] |
| `travel_14d` | +0.274 | -0.003612 | [-0.005790, -0.001481] |
| `home_only` | +0.482 | -0.005921 | [-0.008566, -0.003276] |

14-day distance beyond geography-only flags: loss change -0.003404, cluster interval [-0.004923, -0.001965].

- all_known: Pearson correlation between extra km and baseline win residual +0.0433, 95% tournament-cluster interval [+0.0193, +0.0678], n=41,102. Negative would support the hypothesized direction; this is not causation.
- first_entry_known: Pearson correlation between extra km and baseline win residual +0.0141, 95% tournament-cluster interval [-0.0139, +0.0440], n=10,061. Negative would support the hypothesized direction; this is not causation.

Validation preferred always using the federation-home proxy, not the previous-tournament itinerary. Any predictive improvement may encode geography/home advantage or country-strength differences; it does not demonstrate that longer real journeys worsen play.
The reused historical holdout and geographic proxies do not establish a durable real-travel effect or profitable bets. Production still has no travel feature.

## Travel Distance Repeat Check

Repeated the fixed country-distance setup with seeds 42, 7 and 2026. Rebuilt the identical 302,931-match cohort at the original cutoff, `2026-11-15T05:54:42.807512+00:00`; newly fetched finals were excluded. The seed-42 validation and previously scored 2026 metrics reproduced exactly. No production files, saved forecasts, bets or prior experiment results changed.

Cutoffs are strictly more than 7/14/21/28 days since the last earlier observed completed match before switching the origin to the federation-home country. The model receives `log1p(km)` for each side and their difference, plus home-origin, missing-origin and at-home flags. Doubles average partner km before the log transform. Inbound distance persists across rounds, and unknown locations remain missing. Distances are country representative-point proxies, not verified travel or residence.

Same fixed LightGBM defaults and observed-v2 ratings as the matched control. Each evaluation year trains on earlier-year results available before the year boundary; early stopping uses an availability-purged training tail. Probability is the two-order average. These are means of three separately scored seeds, not ensemble scores or three independent test sets. Validation uses 90,167 matches from 2024-2025; the reused retrospective 2026 benchmark contains 44,179 matches.

| Setting | Validation Accuracy / Loss | 2026 Accuracy / Loss | Helped? |
| --- | ---: | ---: | --- |
| Existing 44-feature control | 76.212% / 0.478904 | 76.146% / 0.480564 | Control |
| 14-day geography flags, no km | 76.397% / 0.476929 | 76.388% / 0.480072 | Small lower loss |
| Always use federation-home origin | 76.535% / 0.473072 | 76.634% / 0.474613 | Best validation and 2026 loss |
| 7-day cutoff | 76.419% / 0.475153 | 76.538% / 0.476613 | Lower loss; best itinerary cutoff on validation |
| 14-day cutoff | 76.435% / 0.475190 | 76.481% / 0.476992 | Lower loss in all three seeds |
| 21-day cutoff | 76.455% / 0.475647 | 76.479% / 0.477495 | Lower loss in all three seeds |
| 28-day cutoff | 76.439% / 0.475471 | 76.531% / 0.477535 | Lower loss in all three seeds |

The overall validation winner was home-only. Among itinerary cutoffs, 7 days narrowly beat 14 days by just 0.000036 mean validation loss, so this is not strong evidence for a uniquely optimal cutoff. These choices were recorded before scoring the additional 2026 variants; that benchmark was already examined in earlier work and is not fresh independent confirmation.

Primary 14-day comparison versus the matched control: mean accuracy +0.335 percentage points and mean log loss -0.003573. Paired 2026 loss changes with 2,000 tournament-year bootstrap resamples (213 clusters):

| Seed | Log-Loss Change | 95% Cluster Interval |
| --- | ---: | --- |
| 42 | -0.003612 | [-0.005790, -0.001481] |
| 7 | -0.003927 | [-0.006249, -0.001755] |
| 2026 | -0.003179 | [-0.005362, -0.001193] |

Home-only improved mean accuracy by 0.489 percentage points and loss by 0.005952; its paired loss intervals also excluded zero in all seeds. Adding 14-day distance beyond the geography flags improved loss in all seeds, but the accuracy difference beyond those flags was not consistently positive.

The correlation between extra km and baseline win residual remained positive, about +0.0433 overall and +0.0141 to +0.0143 at first tournament entry. The original first-entry cluster interval included zero. This does not support the proposed negative fatigue relationship; geographic/home advantage or player-selection effects may explain the predictive signal. No betting-profit claim or production promotion follows from this test.

Compact protocol, per-fold/per-seed metrics, paired intervals, scripts and integrity checks: `outputs/experiments/travel-distance-confirmation-2026-11-15/`. No models, feature matrices or new per-match prediction arrays were retained. SHA-256 verification covered all 15,121 pre-existing protected files before this append-only documentation update; all private paper-bet inputs and settlements were preserved.

## Home Advantage and Country Strength Controls

Isolated factorial follow-up to the home-country distance result. Same frozen cutoff and 302,931 matches as the travel studies; validation 2024-2025 (90,167 matches), reused retrospective 2026 benchmark (44,179). Seeds 42, 7 and 2026, fixed LightGBM defaults, availability-purged inner early stopping and earlier-year refitting. Every raw 44-feature control reproduced the prior study exactly. Scores below are means of separately scored seeds, not ensembles or independent test sets.

Definitions were fixed before training:

- All eight factorial variants share `host_known` and the two represented-country-location-known flags, so distance cannot gain merely from additional missingness information. The original raw control is also reported separately.
- Home advantage: each side's represented-country geographic code equals the host's code (1/0, missing when unknown), plus the side difference. UK subdivisions share GB geographically, as in the distance studies.
- Country controls: country Elo by discipline, starting at 1500, uniform K=20, `E1=1/(1+10^((R2-R1)/400))`, `delta=20*(label-E1)`, then `R1+=delta` and `R2-=delta`. No decay, round/tier/score weighting, domestic matches or unknown-country updates. Features are each side's rating and `log1p(international match count)`, plus differences and country identity categories. ENG/SCO/WAL remain separate rating identities. Categories are learned only from the current training split; unseen test countries are missing categories.
- Distance: `log1p(km)` from represented-country point to host-country point, each side and difference. This is always-home distance, not a prior-tournament itinerary or verified residence/travel. Team-country proxy is used for doubles. All locations are the same cached country representative points used previously.

Country ratings receive only completed eligible results with `available_at < target prediction_time`, in deterministic availability order. Tied predictions cannot share outcomes. Both opponent orders are averaged. The primary contrast was fixed as adding only the three distance columns to the model already containing home advantage and country controls.

| Setting | Validation Accuracy / Loss | 2026 Accuracy / Loss | Finding |
| --- | ---: | ---: | --- |
| Raw 44-feature control | 76.212% / 0.478904 | 76.146% / 0.480564 | Reproducibility anchor |
| Matched baseline with location-coverage flags | 76.282% / 0.478496 | 76.182% / 0.480284 | Factorial control |
| Home advantage | 76.393% / 0.476905 | 76.412% / 0.480147 | Small/mixed loss gain |
| Country controls (Elo + country identity) | 77.922% / 0.450114 | 77.667% / 0.456250 | Large gain in both periods |
| Home advantage + country controls | 77.982% / 0.449616 | 77.578% / 0.456214 | Best validation loss |
| Distance | 76.523% / 0.473534 | 76.653% / 0.474692 | Helps simpler baseline |
| Home advantage + distance | 76.565% / 0.473292 | 76.650% / 0.474725 | Helps simpler baseline |
| Country controls + distance | 77.986% / 0.449786 | 77.649% / 0.455397 | Small mean loss gain beyond country controls |
| Home advantage + country controls + distance | 77.980% / 0.450073 | 77.714% / 0.455600 | Mixed incremental distance evidence |

Primary distance increment beyond both controls: validation mean loss **+0.000457** (worse), accuracy -0.002 percentage points; 2026 mean loss **-0.000614** (better), accuracy +0.136 percentage points. Per-seed paired loss changes, candidate minus controls, with 2,000 tournament-year bootstrap resamples (474 validation clusters; 213 in 2026):

| Seed | Validation Change / 95% Interval | 2026 Change / 95% Interval |
| --- | --- | --- |
| 42 | +0.000621 / [+0.000110, +0.001112] | -0.000739 / [-0.001276, -0.000188] |
| 7 | -0.000635 / [-0.001110, -0.000189] | -0.001177 / [-0.001837, -0.000553] |
| 2026 | +0.001387 / [+0.000832, +0.001942] | +0.000073 / [-0.000548, +0.000710] |

Country controls improve 2026 mean accuracy by 1.484 percentage points and loss by 0.024033 versus the matched baseline; all three paired loss intervals exclude zero. Home advantage alone improves mean loss by only 0.000137, and every per-seed loss interval includes zero. The combined country factor contains Elo and country identity, so this study does not identify which component causes the gain.

Conclusion: country-level information explains most of the predictive improvement. Home-country distance retains a small 2026 gain beyond both controls, but validation and seed results are inconsistent. That is not robust enough to promote distance on this evidence, and it does not establish real travel fatigue or betting profitability. Country controls warrant a separate prospective/rolling evaluation before any production change. No features, models, forecasts or bets were promoted or regenerated.

Compact evidence: `outputs/experiments/home-distance-controls-2026-11-15/` contains protocol, source references, per-seed/per-fold metrics, all paired contrasts, scripts, 10 focused tests and integrity checks. No models, feature matrices or per-match probability arrays were retained. All 15,132 pre-existing protected files were unchanged before this append-only documentation update; production/default bundles, saved predictions and all 84 existing paper-bet records stayed unchanged.

## Country Elo vs Country Identity

Separated the earlier country-control block into ratings, categorical identity and historical match counts. Same frozen cutoff `2026-11-15T05:54:42.807512+00:00`, 302,931 matches, observed-v2 timing, fixed LightGBM defaults and seeds 42, 7, 2026. Validation is pooled 2024-2025 (90,167 matches); the reused retrospective 2026 benchmark has 44,179 matches. Home advantage and distance are off throughout. All variants share the same location-coverage flags. Mean scores below are separate-seed averages, not ensembles or independent test sets.

Exact components:

- Elo only: each side's country Elo and their difference; **no country identity or counts**. Initial 1500, uniform K=20, separate disciplines, logistic expectation with scale 400, no decay/round/tier/score weighting; skip domestic and unknown-country results. Update only from eligible completed international results with `available_at < prediction_time`.
- Identity only: the two represented-country categorical codes; **no country Elo or counts**. Shared country vocabulary across sides is fitted only on each training split, including the purged inner early-stopping fit. Unseen evaluation countries become missing categories.
- Counts only: `log1p` counts of earlier available international matches for each country/discipline and their difference; **no ratings or identity**. This was separated because the previous Elo group also included counts.
- Elo + identity excludes counts. The original combined group restores all three components in exactly the previous column order. Baseline and original-combined metrics, tree counts and sample counts reproduced exactly in every seed/year.

Training labels must be available before each evaluation-year boundary. Early stopping uses an availability-purged earlier training tail, then the model is refit on the available earlier-year rows. Both opponent orders are averaged. No feature definitions or hyperparameters were tuned on these comparisons.

| Setting | Validation Accuracy / Loss | 2026 Accuracy / Loss | Finding |
| --- | ---: | ---: | --- |
| Matched baseline | 76.282% / 0.478496 | 76.182% / 0.480284 | Control |
| Counts only | 76.891% / 0.470019 | 76.734% / 0.471986 | Helps alone, less than either country component |
| Country Elo only | 77.460% / 0.458511 | 76.976% / 0.464784 | Helps alone, weaker than identity |
| Country identity only | 77.854% / 0.451009 | 77.638% / 0.457290 | Strongest individual component |
| Country Elo + identity, no counts | 77.941% / 0.450793 | 77.618% / 0.456986 | Small incremental Elo benefit in mean loss |
| Original combined: Elo + identity + counts | 77.922% / 0.450114 | 77.667% / 0.456250 | Best mean loss, additional benefit is seed-sensitive |

Identity alone improves 2026 mean accuracy by 1.455 percentage points and log loss by 0.022994 versus baseline. Elo alone improves accuracy by 0.794 points and loss by 0.015500; counts alone improve accuracy by 0.552 points and loss by 0.008298. These standalone gains are not additive.

Primary incremental contrasts: adding identity to Elo lowers mean loss by 0.007719 in validation and 0.007798 in 2026; every per-seed paired 95% tournament-cluster interval excludes zero in both periods. Adding Elo to identity lowers mean loss by only 0.000216 in validation and 0.000303 in 2026. All three validation intervals include zero, and only one 2026 seed has an interval excluding zero; 2026 mean accuracy falls by 0.020 percentage points.

Adding Elo beyond identity, paired loss change (candidate minus identity-only), with 2,000 tournament-year bootstrap resamples (474 validation clusters, 213 in 2026):

| Seed | Validation Change / 95% Interval | 2026 Change / 95% Interval |
| --- | --- | --- |
| 42 | -0.000114 / [-0.000743, +0.000537] | -0.000937 / [-0.001696, -0.000170] |
| 7 | -0.000513 / [-0.001083, +0.000094] | +0.000025 / [-0.000738, +0.000772] |
| 2026 | -0.000021 / [-0.000615, +0.000615] | +0.000003 / [-0.000697, +0.000709] |

Counts beyond Elo + identity lower mean loss by 0.000678 in validation and 0.000736 in 2026. The 2026 loss change is +0.000578 for seed 42, -0.001222 for seed 7 and -0.001564 for seed 2026: useful on average, not uniformly helpful. The original combined group still has the lowest mean loss, so this is not evidence that removing every other country feature necessarily improves predictions.

Conclusion: country identity is the clearest and largest individual signal. Elo helps on its own in this backtest but is largely redundant once identity is present in this setup; its incremental evidence is weak. Counts carry additional activity information, with smaller and seed-sensitive marginal gains. These reused historical benchmarks are screening evidence, not a new independent holdout, causal country effects or betting-profit evidence. No production candidate was promoted.

Compact evidence: `outputs/experiments/country-components-2026-11-16/` stores the protocol, exact feature sets, per-fold/per-seed scores, all paired contrasts, six component tests, reused timing/category safeguards and integrity checks. The earlier research helpers were reused only in this process, without modifying their files. No models, feature matrices or per-match prediction arrays were retained. All 15,144 pre-existing protected files were unchanged before this append-only documentation update; production/default models, saved predictions and all 84 paper-bet records stayed unchanged.

## Country Bundle Promotion Check

Direct comparison of exactly two fixed candidates: **country Elo + identity + international-history counts**, versus **the same bundle + home advantage + home-country distance**. Both include the same base player features and location-coverage flags. The expanded package adds only each side's home indicator and `log1p(country-point km)` plus their differences. This is always-home country distance, not actual travel or a previous-tournament itinerary. Definitions, country Elo K=20/initial 1500, training-only country categories and observed-v2 timing are unchanged from the earlier studies.

Five seeds fixed before scoring: **42, 7, 2026, 13, 99**. Same frozen cutoff `2026-11-15T05:54:42.807512+00:00` and 302,931-match cohort. Fixed LightGBM defaults; internal early stopping uses an availability-purged training tail, then refits on available training rows. Both opponent orders are averaged, using raw probabilities as in the current round-forecast workflow. All original three-seed annual sample/tree/accuracy/loss checks reproduced exactly for both bundles.

Annual evaluation trains strictly on earlier-year prediction times and labels available before each year boundary. Pooled 2024-2025 validation has 90,167 matches; annual 2026 has 44,179. The primary operational replay retrains at January/April/July/October 2026 boundaries and scores the next quarter. Q1 reuses the identical January-trained annual predictions; the other quarters refit. Quarter counts are 8,295 / 10,669 / 15,211 / 10,004, covering exactly the same 44,179 fixtures once. There were 60 evaluation-model fits, each with internal early stopping/refitting; no parameter tuning or best-seed selection.

All scores are means of separately scored seeds, not ensemble predictions. Paired uncertainty uses 5,000 tournament-edition/year bootstrap resamples (474 validation clusters; 213 for 2026). Crossed intervals additionally resample the five seeds while sharing each tournament draw. Match counts are not multiplied by the number of seeds. Compact per-seed/per-cluster metric sums are retained so these checks do not require large per-match probability files.

| Evaluation | Bundle | Accuracy | Log Loss | Brier |
| --- | --- | ---: | ---: | ---: |
| 2024-2025 annual validation | Country only | 77.936% | 0.450125 | 0.148249 |
| 2024-2025 annual validation | Country + home + distance | 77.970% | 0.449990 | 0.148184 |
| 2026 annual | Country only | 77.668% | 0.456492 | 0.150624 |
| 2026 annual | Country + home + distance | 77.683% | 0.455624 | 0.150376 |
| **2026 quarterly replay** | **Country only** | **77.741%** | **0.454972** | **0.150131** |
| **2026 quarterly replay** | **Country + home + distance** | **77.716%** | **0.454685** | **0.150043** |

Expanded minus country-only, with crossed seed/tournament 95% intervals. Accuracy differences are percentage points; negative loss differences are better.

| Evaluation | Accuracy Change / 95% Interval | Log-Loss Change / 95% Interval |
| --- | --- | --- |
| 2024-2025 validation | +0.034 / [-0.061, +0.130] | -0.000136 / [-0.000879, +0.000612] |
| 2026 annual | +0.015 / [-0.119, +0.146] | -0.000868 / [-0.001591, -0.000124] |
| **2026 quarterly replay** | **-0.026 / [-0.168, +0.106]** | **-0.000287 / [-0.000865, +0.000294]** |

The earlier three-seed annual accuracy gain of +0.048 points shrank to +0.015 with five seeds. In quarterly replay, only two of five seeds improve accuracy: changes are +0.020 / -0.038 / +0.113 / -0.174 / -0.050 points for seeds 42 / 7 / 2026 / 13 / 99. The mean is about 11.4 fewer correct predictions per 44,179 matches, not evidence that the package is definitively worse because its interval includes both signs. Quarterly mean loss improves in all five seeds, but its tournament/seed interval still includes possible worsening. Brier improves slightly on average. Descriptive 10-bin ECE is 0.005908 vs 0.005801 in quarterly replay; ECE was not a selection gate or recalibration step.

Predeclared historical gates: nonnegative validation accuracy change and nonpositive validation loss change; nonnegative annual-2026 accuracy change; primary quarterly accuracy crossed-interval lower bound strictly above zero; quarterly loss crossed-interval upper bound at most zero; and nonpositive quarterly mean Brier change. The first three and Brier gates pass. **The quarterly accuracy and log-loss confidence gates are not met.** Thresholds, seeds and features were not changed after scoring.

Forward audit: no verified pre-result forecasts or prior exported model bundles match these two exact candidate schemas. Latest local result observation is `2026-11-15T16:06:00.961766Z`, before the original country/home/distance study began. The existing saved finals forecasts use the older production model and cannot be relabeled as candidate forecasts. Annual results, quarterly replay and additional seeds all reuse known historical outcomes, so the separate prospective-forward gate remains **not available**. No new results were fetched for this check.

**Outcome: the expanded package does not pass this promotion check.** It retains a small probability-quality benefit in the historical averages, but no reliable accuracy improvement under the primary quarterly replay. That is not proof that home/distance can never help. Keep both bundles experimental and obtain genuinely pre-result candidate forecasts on new matches before reconsidering a production switch. Existing production was not changed, and neither candidate was promoted.

Evidence: `outputs/experiments/country-home-distance-promotion-2026-11-16/` contains the locked protocol, forward audit, fold/seed scores, paired intervals, compact cluster sums, scripts and 10 focused check tests. The reused country timing/category safeguards also pass. One initial attempt stopped at metric reporting because of a missing calibration-report import; only the isolated script was fixed, the attempt metadata is retained, and the gate definitions were verified unchanged. No model bundles, feature matrices or per-match probability arrays were retained. All 15,156 pre-existing protected files stayed unchanged before this append-only documentation update; production/default bundles, saved predictions and all 84 paper-bet records remain unchanged.

## Country Candidate vs Active Production

Direct research-only promotion check against the active bundle selected by `outputs/forecast_bundle.json`: `outputs/runs/2026-11-12-observed-timing-v2-054212Z`. Production uses the original 44 player/context features, fixed-default LightGBM parameters and raw two-order probabilities. The unchanged 55-feature candidate adds country Elo and history-count side values/differences, two country categories, and the same three location-coverage flags used in the earlier country studies. Home indicators and distance are excluded. The gain therefore describes this complete existing candidate, not a causal attribution to a single country feature.

The current saved artifact was refit on 302,429 known rows through a `2026-11-12T05:42:12` history snapshot. It cannot honestly be scored on all earlier 2026 matches as an out-of-sample model. The primary comparison instead retrains the **production recipe** and candidate at identical historical cutoffs, using the active model's actual hyperparameters. Execution threads are fixed to four and seeds vary; all other model settings match. Early stopping uses an availability-purged training tail before refitting the available earlier rows. The first three seeds reproduce prior raw-production and country-candidate annual sample counts, trees, accuracy and loss exactly.

Same frozen study cutoff `2026-11-15T05:54:42.807512+00:00`, 302,931-match cohort and seeds **42, 7, 2026, 13, 99**. Annual 2024-2025 validation includes 90,167 matches. Annual 2026 and the primary quarterly-retraining replay evaluate the same 44,179 fixtures. Quarterly models train at January/April/July/October boundaries, with prediction times and label availability strictly before the boundary; Q1 reuses the identical January annual fit. Country history releases results only before each target prediction; country categories are fitted only on the relevant training split. No hyperparameters, probability threshold or calibrator were tuned during this check.

| Evaluation | Bundle | Accuracy | Log Loss | Brier |
| --- | --- | ---: | ---: | ---: |
| 2024-2025 validation | Production recipe | 76.206% | 0.478940 | 0.158751 |
| 2024-2025 validation | Country candidate | 77.936% | 0.450125 | 0.148249 |
| 2026 annual | Production recipe | 76.150% | 0.480512 | 0.159423 |
| 2026 annual | Country candidate | 77.668% | 0.456492 | 0.150624 |
| **2026 quarterly replay** | **Production recipe** | **76.234%** | **0.480095** | **0.159279** |
| **2026 quarterly replay** | **Country candidate** | **77.741%** | **0.454972** | **0.150131** |

Scores are averages of separately scored seeds, not ensemble predictions. Paired intervals use 5,000 tournament-edition/year bootstrap draws, crossed with seed resampling; seeds are not counted as independent extra matches. There are 474 validation clusters and 213 in 2026. Candidate minus production recipe:

| Evaluation | Accuracy Change (pp) / Crossed 95% Interval | Log-Loss Change / Crossed 95% Interval |
| --- | --- | --- |
| Validation | +1.730 / [+1.398, +2.083] | -0.028815 / [-0.033578, -0.024516] |
| 2026 annual | +1.518 / [+1.080, +1.978] | -0.024020 / [-0.029663, -0.018795] |
| **2026 quarterly replay** | **+1.508 / [+1.063, +1.982]** | **-0.025123 / [-0.030761, -0.019892]** |

The primary mean gain is 666 additional correct predictions per 44,179 matches, averaged over five seeds. All seeds improve accuracy, by +1.455 / +1.646 / +1.390 / +1.578 / +1.469 percentage points respectively. Primary Brier change is -0.009148, crossed interval [-0.011374, -0.007167]. **Every predeclared historical gate passes**: validation accuracy/loss, annual accuracy, strictly positive primary accuracy interval, nonpositive primary loss interval upper bound, and non-worsening mean Brier. Descriptive ECE is mixed: primary ECE is 0.005037 for the production recipe and 0.005908 for the candidate despite better proper scores. Do not describe this as uniformly improved calibration; no recalibration was performed.

### Actual Saved-Model Diagnostic

A separate comparison uses the actual saved production weights, never its old training matches. The original saved receipt roster alone initially reconstructs 302,645 rows because current raw files also contain 216 older matches ingested later. Using the historical observation map at the original snapshot excludes those later arrivals and recovers the exact 302,429-row training roster. The recreated production booster dump is **identical**, with maximum probability difference **0.0**. The candidate then trains on that same snapshot with the original model's 500 trees, seed and parameters. This repair and its audit protocol do not alter the primary replay, thresholds or candidate definition.

Only 249 target prediction times strictly after the production history snapshot are scored, with history refreshed before each target. These are a small, already-seen subset of the 2026 cohort, not additional independent cases, a genuine forward test, or a regeneration of the frozen betting forecasts.

| Post-Snapshot Diagnostic | Accuracy | Log Loss | Brier |
| --- | ---: | ---: | ---: |
| Actual saved production artifact | 74.699% | 0.483388 | 0.161053 |
| Country candidate, same training snapshot | 74.699% | 0.477536 | 0.158790 |

Both correctly predict 186 of 249 matches. Candidate loss change is -0.005852, paired 95% interval [-0.013615, +0.013082]; accuracy change is zero, interval [-1.667, +0.680] percentage points. Only three tournament clusters are represented, so this is an inconclusive small diagnostic, not a contradiction of or independent confirmation for the broader replay. Diagnostic ECE also worsens (0.022961 to 0.038005).

**Outcome: the country candidate passes the historical promotion gates against the active production recipe.** This is substantially stronger evidence than the earlier home/distance increment. The small actual-artifact diagnostic does not yet establish an accuracy improvement, and verified pre-result country-candidate forecasts are absent, so the prospective-forward gate remains unavailable. It is a strong candidate for controlled forward validation, not proof of future profitability or an automatic production switch. No candidate model bundle was exported or promoted.

Evidence: `outputs/experiments/country-vs-production-promotion-2026-11-16/` stores the production reference hashes/parameters, locked protocol, forward audit, annual/quarterly scores, crossed paired intervals and compact cluster sums, plus the separately locked artifact reconstruction audit. Primary replay scores/gates are hash-verified unchanged by the artifact follow-up. All 15,174 pre-existing protected files stayed unchanged before this append-only documentation update; production/default bundles, all saved predictions/odds and all 84 paper-bet records remain unchanged.

## Country Default Promotion (2026-11-16)

Following explicit approval, new forecasts now select the separate `outputs/runs/2026-11-16-country-context-181718Z` bundle. It adds country Elo + identity + counts and the same three coverage indicators used in the promotion check: **55 features total**, with **no home advantage or distance**. Country Elo is per discipline, starts at 1500, uses uniform K=20 and excludes domestic/unknown-country results. Counts are `log1p(intercountry matches)`. The 157-country identity vocabulary is fitted only on the final training roster and frozen for inference; unknown identities become missing values. The cached reference registry is embedded, so forecasts do not depend on experiment folders.

The new all-data refit uses **302,941 eligible matches**, a fixed `2026-11-16T18:17:18Z` cutoff, the original model's seed 42 and 500-tree parameters, and raw two-order-averaged probabilities. The latest included result receipt is `2026-11-15T16:06:00.961766Z`. Country features match the research replay **exactly for every training row**. The earlier 77.741% accuracy / 0.454972 log loss / 0.150131 Brier values remain five-seed historical quarterly-replay evidence, not an evaluation of this all-data refit. Genuine prospective validation and profitability remain unproven.

The previous `outputs/runs/2026-11-12-observed-timing-v2-054212Z` model and all its artifacts remain unchanged as the backup; `outputs/forecast_bundle.backup.json` preserves its pointer. The older `outputs/artifacts/` bundle also remains untouched. New forecasts rebuild player/country ratings, counts, recent form and H2H from results available strictly before their frozen cutoff. Historical predictions, quoted odds and all private betting records are not regenerated or re-scored. Old calibrators are not attached to the new feature schema.

Verification: **296 core tests pass**, including new state serialization, strict result release, country-category encoding, opponent-order symmetry, country-history correction/retraction, raw-only bundle loading and legacy compatibility checks. The new run also stores exact feature-parity and integration-check reports. A synthetic end-to-end round forecast uses the real local history refresh without recording bets; temporary synthetic forecast files are removed. Pre-existing data/output files are hash-checked, allowing only the forecast pointer and this task's documentation update to change.

## Receipt Completion and Quarantine Refit (2026-11-16)

New receipts now require and fingerprint explicit final status. Legacy completion proofs are appended only when exact cached response bytes match their original recorded SHA-256 and the normalized result is final. All 609 observed matches retain their original availability times; all original database rows remain intact. The seven audited winner/score contradictions are quarantined by both match ID and fixture signature, without editing raw files or guessing corrected winners. Explicit non-final results and duplicate quarantine aliases cannot enter training/history.

Matched comparison: the active 55-feature country recipe with its pre-fix data versus the same recipe with cleaned data. Five seeds (42/7/2026/13/99); January/April/July/October retraining, availability-purged internal early stopping, frozen training-only country vocabularies and raw two-order averaging. No new tuning. Both variants score the same 44,188 non-quarantined 2026 fixtures; the bad 2026 target is excluded from both sides.

| Quarterly Replay | Matches | Current Accuracy / Loss | Cleaned Accuracy / Loss |
| --- | ---: | ---: | ---: |
| All 2026 | 44,188 | 77.739% / 0.455025 | 77.727% / 0.455410 |
| Senior World Tour | 4,559 | 72.779% / 0.526686 | 72.814% / 0.526814 |
| World Tour R32 | 2,106 | 74.340% / 0.502224 | 74.311% / 0.503142 |
| Junior tagged | 18,023 | 78.047% / 0.455776 | 78.011% / 0.456358 |

Pooled cleaned-minus-current accuracy is **-0.0118 percentage points**, crossed seed/tournament 95% interval **[-0.1086, +0.0755]**. Log-loss change is +0.000384, interval [-0.000027, +0.000829]; Brier change is +0.000129, interval [-0.000025, +0.000292]. All include zero. Secondary annual accuracy is 77.665% current versus 77.624% cleaned. The defects are fixed, but this reused historical comparison does not establish an accuracy or probability-quality gain, prospective superiority or profit.

A separate 500-tree seed-42 candidate was saved at `outputs/runs/2026-11-16-receipt-quality-cleaned`, refit on **302,934** clean matches at the same `2026-11-16T18:17:18Z` data cutoff as production. Neither all-data artifact is scored on its own historical training rows. At the comparison stage, production weights/state/default pointer and all 16 original Hong Kong run files, original odds and paper-bet accounting remained byte-for-byte unchanged. A new 79-fixture shadow-probability file is stored separately and does not reselect or record bets. The candidate was **not promoted during the comparison**; the later approved promotion is recorded below.

Verification: **312 tests pass**. All 15,206 pre-existing data/output files are checked, allowing only the evidence-backed database migration and data instructions to change; all raw result files and original database rows are preserved. Full [comparison and remediation report](outputs/experiments/receipt-quality-retrain-2026-11-16/report.md), locked protocol, per-fit checkpoints, cohort evidence, migration backup and candidate verification are under the new experiment directory. Large data/model artifacts remain local.

## Cleaned Default Promotion (2026-11-16)

Following explicit user approval, `outputs/forecast_bundle.json` now selects `outputs/runs/2026-11-16-receipt-quality-cleaned` for new forecasts. This is the already verified 302,934-row, 55-feature, seed-42 500-tree refit; no new training, tuning or feature selection is performed during promotion. Historical evidence remains the matched 77.739% current-control versus 77.727% cleaned quarterly replay, not proof of a gain or prospective profitability.

`outputs/forecast_bundle.previous.json` preserves the previous country bundle. The older observed-v2 backup pointer and all existing model binaries remain unchanged. All 16 original Hong Kong files, their supplied odds, and the structured/rendered paper-bet ledger remain byte-for-byte frozen. The private betting records are not regenerated or reselected. Promotion verification, selector provenance and synthetic default-forecast checks are recorded separately from the earlier comparison-stage reports.
