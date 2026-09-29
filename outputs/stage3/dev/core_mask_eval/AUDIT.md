# Stage3 dev Core-mask retrieval audit

Status: COMPLETE, validation PASS. This is dev configuration selection only.
No final eval cohort was constructed or examined. All p-values below are
descriptive dev results, not final-paper hypothesis tests.

## Scope and frozen protocol

- Input dev cohort has 50 places disjoint from Stage2-100. The available scene
  and family audit covers its first 20 queries; the other 30 remain unevaluated.
- Candidate selection is deterministic, before model loading, with no access to
  retrieval outcomes: highest full-map continuous weighted coverage, highest
  target precision, smallest area, smallest centroid distance, candidate ID.
- Target precision remains 0.7. All weighted thresholds 0.3/0.5/0.7 are retained.
  Empty arms are explicit, never replaced with lower-threshold masks.
- The frozen BoQ checkpoint and all 127,118 SUPPORT references are used.
  Every clean descriptor and full-SUPPORT margin exactly matches the prior
  vulnerability export (maximum absolute error 0).
- Intervention occurs at original SOURCE resolution. Core pixels receive the
  original image's RGB channel means, rounded to uint8 before the unchanged
  Stage2 uint8 resize/normalization. Requested/applied fills are logged. This is
  a native-pixel intervention, distinct from Stage2's 224-space intervention.
- Five random controls per mask are exact integer translations of its native
  footprint. All 100 saved random masks pass coordinate, area and connectivity
  checks. All 20 masks in this run received five distinct legal locations,
  although the preregistered Stage2 sampling rule permits replacement.
- Random margin drops are averaged within each query. All statistics count
  queries, never individual random repeats. Bootstrap uses 20,000 paired query
  resamples, seed 0, percentile 95% CI. Wilcoxon is two-sided asymptotic.

## Results

Primary advantage = targeted margin drop − mean random margin drop.

| Arm | Paired queries | Mean target drop | Mean random drop | Mean advantage | Median advantage | 95% bootstrap CI | Wilcoxon p (dev only) | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Family, weighted ≥0.3 | 2 | 0.069025 | 0.049688 | 0.019337 | 0.019337 | [0.001101, 0.037573] | 0.179712 | 2/2 |
| Family, weighted ≥0.5 | 0 | — | — | — | — | — | — | — |
| Family, weighted ≥0.7 | 0 | — | — | — | — | — | — | — |
| Old 6% ellipse | 18 | 0.036532 | 0.011780 | 0.024752 | 0.023638 | [0.010378, 0.038957] | 0.012275 | 13/18 |

Both eligible family queries select `parked_vehicle`, approximately 12% area:

| Place | Target drop | Random mean drop | Paired advantage |
| --- | ---: | ---: | ---: |
| Osaka:3358 | 0.072524 | 0.034951 | 0.037573 |
| PRS:3993 | 0.065525 | 0.064424 | 0.001101 |

These are 2/20 audited queries (10%), not evidence that the method covers all 50
dev queries. The positive bootstrap interval with n=2 reflects an extremely
limited empirical distribution; it is not a robust population claim.

The ellipse adapter failed its unchanged compactness/overlap search on
`Lisbon:6940` and `Boston:1662`. Their failures are recorded, not substituted.
On the two common queries, family advantage minus ellipse advantage averages
**−0.004362**, 95% CI [−0.009832, 0.001108], win rate 1/2. Whole-arm means use
different eligible subsets and must not be interpreted as a head-to-head gain.
The family and ellipse footprints also have different area budgets.

## Dev freeze and artifacts

The rule declared before model loading prefers the family threshold with most
eligible paired queries (minimum two), then mean advantage, then higher
threshold. It selects **0.3**, the only eligible arm, and saves the complete
configuration and hashes in [frozen_config.json](frozen_config.json). This is a
dev freeze for the current candidate configuration; n=2 is a substantial
limitation, and no final-paper superiority claim is supported.

Numerical artifacts: [per-condition CSV](per_condition.csv),
[query-paired CSV](paired.csv), [statistics](statistics.csv),
[summary](summary.json), [protocol](protocol.json),
[selected masks](chosen_core_masks.json),
[translations](random_placements.json), [exclusions](exclusions.json),
[common-query comparison](ellipse_common_queries.csv).

[Validation](validation.json) independently verified all 20 selected masks,
100 random translations, 140 per-condition rows, 20 query/arm pairs, all arm
statistics, 66 input hashes and 70 protected Stage2/analysis/cohort files.
The four Stage3 test modules passed all 60 tests. No Stage1/Stage2 file or
threshold changed; no VLM, Diffusion, vulnerability mining or training ran.
