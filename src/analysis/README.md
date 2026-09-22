# Attention Proposal Map and stage-1 occlusion

These utilities analyze a frozen BoQ model. They do not train a model or construct
a causal map. `train.py`, model parameters, and default forward behavior are unchanged.

## Protocol

- Decode RGB, bicubic resize on uint8, convert to float32 `[0,1]`.
- Extract clean attention after ImageNet Normalize. Read the grid from the actual
  backbone output. Mean heads, then queries, then layers.
- Rank **raw** token attention. Percentile clipping is only for display.
- Compare `attention_connected` (8-neighbor frontier growth from the maximum)
  with `attention_raw_topk` (globally highest scores). Both select exactly
  `round(ratio * Ht * Wt)` tokens from the **same** aggregated attention map.
- Primary random baseline is now **`random_token`**: translate the original token
  mask to other legal grid positions, THEN nearest-upsample. Sample without
  replacement if at least five alternatives exist. Otherwise use all alternatives
  once in random order, then sample the remaining draws with replacement.
- Retain **`random_pixel`** as the original integer-pixel translation baseline,
  sampled with replacement. Both baselines exclude the original location but may
  overlap with the targeted region. Each mask construction has its OWN
  shape-matched baselines; never use a connected shape to control raw-topk.
- For strict pixel equality, the runner requires RGB dimensions divisible by the
  actual token grid. Every nearest cell then has identical area, and token
  translations preserve topology, alignment, shape, and pixel count exactly.
  Nonuniform grids fail explicitly rather than silently introducing area changes.
- If no alternative placement exists, keep the target for the direct mask
  construction comparison. Exclude it only from its affected mask/baseline paired
  cohort. In particular, scattered raw-topk masks can span the full image and
  cannot legally translate. Report this; do not crop or replace their shapes.
- Ratios remain 10%, **15%**, 20%. Operator, clean query, checkpoint, database,
  and mean-head -> mean-query -> mean-layer aggregation are identical throughout.
- Primary fill is the original RGB image's channel mean, computed over the whole
  clean resized image. All placements use that same fill. Normalize **after**
  erasing. `imagenet_mean` and `gaussian_blur` are explicit secondary options.
- Keep the full database clean. Cache its descriptors once, checking checkpoint,
  preprocessing, ordered reference manifest, model source and inference settings.
- Use five random draws/query. First average each endpoint within query, then
  compare targeted vs random across paired queries. Never pool the draws as
  independent observations. Primary endpoint: **margin_drop**.
- Similarity is dot product of normalized descriptors. Rank uses the best positive
  among the full ground-truth set. Hardest negative searches the full remaining
  database. Exact similarity ties use ascending reference index.

## Running

In the `boq` environment, from the repository root:

```bash
python scripts/eval_targeted_occlusion.py \
  --checkpoint /path/to/model.ckpt --backbone dinov2_vitb14 \
  --dataset msls-val --dataset-path data/val/msls-val --image-size 322 322 \
  --output-dir outputs/stage1/ablation100 \
  --reference-cache outputs/stage1/reference_descriptors.pt \
  --num-queries 100 --random-repeats 5 --seed 2024 --device cuda \
  --mask-modes connected_topk raw_topk \
  --random-baselines random_pixel random_token --primary-baseline random_token
```

After inspecting correctness, use the same arguments with `--num-queries 500`
and `--output-dir outputs/stage1/ablation500`. The query subset is a seeded permutation
prefix, so the 500-query run includes the initial 100. Cache identity mismatch is
an error; select another cache path rather than silently using stale descriptors.
Do not expand beyond 500 without a trend. There is no automatic full-dataset run.

Additional plotting dependencies are `matplotlib` and `PyYAML` (present in the
local `boq` environment). Run regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Outputs and interpretation

Each run contains `config.yaml`, `clean_metrics.json`, `per_query.csv`,
`paired_query.csv`, `summary.csv`, `ablation_summary.csv`, and
`ablation_paired_query.csv`. `comparisons/<mask_mode>/<baseline>/visualizations/`
contains the five diagnostic plots per pairing. Root `visualizations/` contains
direct ablation plots and reservoir-sampled primary-baseline cases.
The shared `reference_descriptors.pt` is named in
the configuration. Clean metrics use all selected queries. Summary/plot conditions
use identical eligible queries **within each ratio**; report cohort size when
comparing ratios. `mask_area_actual` is a fraction of resized RGB pixels, and
`mask_pixels` is the integer count. `random_repeat=-1` denotes a non-random row.

The CSV schema is version 2. Conditions are `clean`, `attention_connected`,
`attention_raw_topk`, `random_pixel`, and `random_token`. Always group random rows
by **mask_mode AND condition**, as well as query and ratio. `shift_y/x` remains in
pixels; `token_shift_y/x` records the grid offset for token translations. For
misaligned `random_pixel`, `mask_tokens` is empty; `source_mask_tokens` records
the originating attention-mask budget. Sampling policy, unique placements, and
eligibility for each baseline are explicit columns.

`paired_query.csv` contains one row per query/ratio/mask_mode/baseline, including
the mean of five random draws. `paired_margin_difference = targeted margin_drop - mean random
margin_drop`; positive values favor the hypothesis. Summary confidence intervals
resample **queries**, not random draws. They are exploratory query-bootstrap
intervals, not a correction for multiple ratios or sequential examination.

`ablation_summary.csv` directly compares `attention_raw_topk - attention_connected`
on ALL selected queries (including untranslatable masks). It also compares
`random_token - random_pixel` within each construction on the intersection of
their eligible queries. These are paired contrasts, not differences between
averages of unequal cohorts. Pixel and token baselines also differ in their
requested sampling policy (with versus usually without replacement); interpret
the baseline contrast as this specified protocol comparison.

Failure flip is `clean R@1 correct AND perturbed R@1 incorrect`. The main rate uses
all paired queries as denominator; the conditional rate among clean-correct
queries is additionally reported. A random hit/flip rate is the within-query draw
mean, not the retrieval result of an averaged descriptor.

Cases use the primary ratio and a configurable absolute paired margin difference
threshold (default 0.01), with reservoir sampling within `attention_stronger`,
`random_stronger`, and `almost_identical`. A shown random image is draw 0; captions
and JSON separately identify its rank and the mean damage across all draws.
These category names do not imply statistical significance.
