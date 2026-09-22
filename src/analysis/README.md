# Attention Proposal Map and stage-1 occlusion

These utilities analyze a frozen BoQ model. They do not train a model or construct
a causal map. `train.py`, model parameters, and default forward behavior are unchanged.

## Protocol

- Decode RGB, bicubic resize on uint8, convert to float32 `[0,1]`.
- Extract clean attention after ImageNet Normalize. Read the grid from the actual
  backbone output. Mean heads, then queries, then layers.
- Rank **raw** token attention. Percentile clipping is only for display.
- Grow `connected_topk` from the maximum using the highest-scoring 8-neighbor
  frontier until exactly `round(ratio * Ht * Wt)` tokens have been selected.
- Expand masks with nearest interpolation. Primary ratios: 10%, **15%**, 20%.
- Random masks are exact integer-pixel translations of that pixel mask, uniformly
  sampled with replacement from legal positions other than the original. Overlap
  with the targeted mask is permitted. Shapes and actual pixel counts are equal.
- If no alternative placement exists, keep the targeted measurement, mark it
  ineligible, and exclude that query from **all** paired conditions at that ratio.
  Report exclusions; the resulting paired cohort need not represent all queries.
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
  --output-dir outputs/stage1/run100 \
  --reference-cache outputs/stage1/reference_descriptors.pt \
  --num-queries 100 --random-repeats 5 --seed 2024 --device cuda
```

After inspecting correctness, use the same arguments with `--num-queries 500`
and `--output-dir outputs/stage1/run500`. The query subset is a seeded permutation
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
`paired_query.csv`, `summary.csv`, and `visualizations/` (five required plots plus
seeded reservoir-sampled cases). The shared `reference_descriptors.pt` is named in
the configuration. Clean metrics use all selected queries. Summary/plot conditions
use identical eligible queries **within each ratio**; report cohort size when
comparing ratios. `mask_area_actual` is a fraction of resized RGB pixels, and
`mask_pixels` is the integer count. `random_repeat=-1` denotes a non-random row.

`paired_query.csv` contains one row per query/ratio, including the mean of five
random draws. `paired_margin_difference = targeted margin_drop - mean random
margin_drop`; positive values favor the hypothesis. Summary confidence intervals
resample **queries**, not random draws. They are exploratory query-bootstrap
intervals, not a correction for multiple ratios or sequential examination.

Failure flip is `clean R@1 correct AND perturbed R@1 incorrect`. The main rate uses
all paired queries as denominator; the conditional rate among clean-correct
queries is additionally reported. A random hit/flip rate is the within-query draw
mean, not the retrieval result of an averaged descriptor.

Cases use the primary ratio and a configurable absolute paired margin difference
threshold (default 0.01), with reservoir sampling within `attention_stronger`,
`random_stronger`, and `almost_identical`. A shown random image is draw 0; captions
and JSON separately identify its rank and the mean damage across all draws.
These category names do not imply statistical significance.
