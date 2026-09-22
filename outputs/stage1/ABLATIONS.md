# Token-aligned random and mask-construction ablations

Both ablations are implemented and were run on the same nested 100/500-query
MSLS-val subsets, seed 2024, local epoch 21 checkpoint, RGB 322x322, float32,
mean fill. The full 18,871-reference database is unchanged. There was no new
training, attention weighting scheme, or causal map.

## Implementation

- `random_pixel`: preserves the earlier uniform integer-pixel translation,
  with replacement, excluding the original position.
- **`random_token` (new default primary)**: translates the original token mask
  on the actual 23x23 grid before nearest upsampling. Five positions are sampled
  without replacement whenever at least five alternatives exist. Otherwise all
  legal alternatives appear once before replacement draws fill the remaining slots.
- `attention_connected`: original maximum-seeded 8-neighbor frontier growth.
- `attention_raw_topk`: highest K global token scores, no connectivity constraint.
- Every construction uses K = round(ratio * Ht * Wt). Each has its OWN random
  controls, preserving its complete shape and topology. The 10%, 15%, 20% masks
  contain respectively 53, 79, 106 tokens. At this resolution, every token covers
  exactly 14x14 pixels, so random/target areas are exactly equal.
- The runner rejects nonuniform nearest-cell configurations for these strict
  ablations, rather than silently allowing pixel-area differences.
- Random endpoints are averaged within query; statistics and bootstrap units are
  queries, never individual draws.

## Primary ratio: 15%, 500 queries

For the connected-mask comparison, all conditions use the same 500 queries:

| Condition | Mean margin drop | Recall@1 |
|---|---:|---:|
| Clean | 0 | 89.20% |
| random_pixel | 0.031444 | 85.64% |
| random_token | 0.027290 | 86.80% |
| attention_connected | 0.071935 | 79.40% |

Connected minus token-random paired margin drop is **+0.044645**, exploratory
query-bootstrap 95% interval **[0.040043, 0.049079]**. The targeted advantage
persists with patch-aligned random locations.

On the same 500 connected-mask queries, token-random minus pixel-random mean
margin drop is -0.004153, interval [-0.006013, -0.002157]. This compares the
specified baseline protocols: alignment AND sampling policy differ (pixel uses
replacement; token normally does not). It does not isolate alignment alone.

## Raw versus connected: direct comparison on all 500 queries

| Construction | Mean margin drop | Recall@1 |
|---|---:|---:|
| attention_connected | 0.071935 | 79.40% |
| attention_raw_topk | 0.072247 | 78.80% |

Raw minus connected paired margin drop is **+0.000312**, interval
**[-0.001071, 0.001678]**. This run does not establish a stable primary-endpoint
advantage for either construction. All queries, budgets, RGB pixel counts,
checkpoint, fill values, normalization, and reference descriptors are matched.

Two raw masks cannot move to any other legal grid/pixel location at 15% because
their bounding boxes span the full image. They remain in the direct construction
comparison above. Their exclusion applies only to raw-mask random-control pairs,
which use 498 queries. On that paired subset, raw-target minus token-random margin
drop is +0.044143, interval [0.039809, 0.048457]. Do not compare its unpaired mean
with a 500-query connected mean as if they had identical cohorts.

Raw-mask random-pair exclusions at 10%, 15%, 20%: 2, 2, 3 queries respectively.
Connected-mask exclusions: zero at all three ratios.

## Validation and reproducibility

- 16 regression tests passed, including disconnected topology preservation,
  exact token/pixel translation, alignment, unique placements, sparse legal
  positions, impossible translations, and separation of statistical cohorts.
- 100-query smoke run completed before the 500-query run. The first 100 queries'
  CSV rows are exactly identical between the two ablation runs.
- Original pixel-baseline sampling was preserved; 100-query shifts and retrieval
  ranks match the previous stage, with margin differences below 1e-6 from changed
  inference batch shapes. No model operations or attention aggregation changed.
- Validated all 500-query token counts, pixel counts, nonzero shifts, multiples
  of 14 pixel offsets, distinct-position counts, and within-query random means.
- Both mask constructions use the same 500 query IDs as the prior stage.
- User-added comments changed the raw model-source cache hashes. A separate
  cache copy was created only after matching the original source hashes and
  verifying identical executable ASTs. Its descriptor tensor is exactly equal
  to the original. The original staged cache was not modified, and references
  were not re-encoded. See `.cache/occlusion_ablations/cache_provenance.json`.
- Bootstrap intervals use 2,000 query resamples. These are exploratory, are not
  corrected for multiple comparisons, and the nested 500-query sample is not
  independent of the earlier 100-query sample.

## Outputs

- [500-query configuration](ablation500/config.yaml)
- [Per-condition CSV](ablation500/per_query.csv)
- [Target-vs-random summaries](ablation500/summary.csv)
- [Target-vs-random paired queries](ablation500/paired_query.csv)
- [Direct ablation summaries](ablation500/ablation_summary.csv)
- [Direct ablation paired queries](ablation500/ablation_paired_query.csv)
- [Raw versus connected plot](ablation500/visualizations/mask_construction.png)
- [Token versus pixel random plot](ablation500/visualizations/connected_topk_random_alignment.png)
- [Connected versus primary token-random plot](ablation500/comparisons/connected_topk/random_token/visualizations/margin_drop_attention_vs_random.png)
- [100-query summaries](ablation100/summary.csv)

The five diagnostic plots for each pairing are in
`ablation500/comparisons/<mask_mode>/<baseline>/visualizations/`.
Seeded case samples against the token baseline are in
`ablation500/visualizations/cases/`. Evaluation stopped at 500 queries.
