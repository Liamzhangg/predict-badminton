# Travel Distance Screening

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
