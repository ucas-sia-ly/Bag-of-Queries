# Stage 1: fair targeted occlusion

The 500-query run shows an exploratory trend supporting H1 for the selected
checkpoint, MSLS-val subset, and mean-fill operator. This is not a causal map
or a claim that raw attention identifies shortcuts.

## Fixed protocol

- Local epoch 21 checkpoint, DINOv2 ViT-B/14 + BoQ, 8192-dimensional descriptors.
- Checkpoint SHA256: `051323a4b683bee4fdfb55ed93e4a2213bdff8720bcce214695c7c145255431d`.
- MSLS-val: all 18,871 references remain clean; their descriptors were computed
  once in run100 and reused by run500.
- 322x322 RGB input, seed 2024, CUDA float32, TF32/xFormers disabled.
- Connected attention masks at 10%, 15%, 20%; primary ratio 15%.
- Original clean RGB channel mean fill, followed by ImageNet Normalize.
- Five uniformly sampled legal integer-pixel translations/query/ratio, excluding
  the original position, sampled with replacement. Same shape and pixel area.
- Random endpoints averaged within each query before paired comparison.
- 100-query correctness run, then nested 500-query run. No full-query-set run.
- No exclusions for unavailable translations in either run, at any ratio.

## Primary results: 500 queries, 15% mask

| Condition | Mean margin drop | R@1 | R@5 | R@10 | Failure flip / all queries |
|---|---:|---:|---:|---:|---:|
| Clean | 0 | 89.20% | 93.80% | 94.80% | 0% |
| Random (within-query mean) | 0.031444 | 85.64% | 92.80% | 94.20% | 4.56% |
| Attention-targeted | 0.071935 | 79.40% | 90.00% | 91.80% | 11.20% |

The mean paired difference in margin drop (targeted minus random) is **+0.040492**,
with exploratory query-bootstrap 95% interval **[0.036184, 0.044862]** (2,000
resamples). This interval treats queries, not the five random draws, as the
independent resampling units. It is not corrected for examining multiple ratios
or for the earlier 100-query look. The 500-query subset includes those first 100
queries and is therefore not an independent replication.

At 10% and 20%, the mean paired differences are respectively +0.034582 and
+0.045002. The initial 100-query primary difference was +0.047008, interval
[0.036977, 0.057894].

## Validation

- 12 regression tests passed: aggregation, normalization, state restoration,
  exact budgets, connectivity, shape-preserving translation, impossible placements,
  RGB fill, full-database rank/similarity metrics, and query-level averaging.
- Verified 9,500 per-condition rows for the 500-query run and 1,500 paired rows.
- Verified all random masks have the exact targeted pixel count, nonzero shifts,
  and IoU below 1 with the targeted mask.
- Verified all rank, hit, flip and margin-damage columns against their definitions.
- Verified the first 100 queries' CSV rows are exactly identical between runs.
- All five plots were generated and visually inspected.
- Reservoir-sampled three cases in each primary-margin category. Category counts:
  attention stronger 355, random stronger 64, almost identical 81. The threshold
  is 0.01 margin units and does not imply statistical significance.

## Artifacts

- [500-query configuration](run500/config.yaml)
- [500-query summary](run500/summary.csv)
- [Per-condition measurements](run500/per_query.csv)
- [Per-query paired measurements](run500/paired_query.csv)
- [100-query summary](run100/summary.csv)
- [Margin drop comparison](run500/visualizations/margin_drop_attention_vs_random.png)
- [Descriptor drift comparison](run500/visualizations/descriptor_drift_attention_vs_random.png)
- [Rank degradation ECDF](run500/visualizations/rank_degradation_ecdf.png)
- [Recall versus mask ratio](run500/visualizations/recall_at_k_vs_mask_ratio.png)
- [Failure flip rate](run500/visualizations/failure_flip_rate.png)
- [Case sampling counts](run500/visualizations/cases/categories.json)

The cache and generated outputs are excluded from Git via `outputs/stage1/`.
