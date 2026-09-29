# Stage3 dev cohort and target export

## Independent dev cohort (schema version 1)

```bash
python scripts/stage3_build_cohort.py --dev 50 --seed 0
python -m pytest -q tests/test_stage3_cohort.py
```

The standard-library-only builder reads the frozen
`outputs/stage2/split/gsv_split.jsonl`, seed0/seed1 cohorts in both
`outputs/stage2/gsv_occlusion/{support,prototype}/config.json`, and
`outputs/stage3/targets/targets.jsonl`. It verifies SOURCE identities against
the split and excludes the **union of all Stage2 cohort places**, including
places absent from the target export. Existing targets outside that union fail
the exposure audit. No masks or retrieval metrics are used for selection.

For the current artifacts, each seed contains 50 distinct places, seed0 and
seed1 do not overlap, and support/prototype use the same 100 places. All 100
existing target places belong to this Stage2-100. Of 63,559 SOURCE places,
63,459 remain eligible for dev.

Selection first orders eligible places by SHA-256 of the UTF-8 string
`stage3-dev-v1:place:{seed}` + NUL + `place_key`, breaking ties by key. The first
`dev` places are selected. Within each, the minimum hash of
`stage3-dev-v1:image:{seed}` + NUL + `image_key` chooses one SOURCE (key breaks
ties). Row order, Python hash randomization, view counts, image content and
model outcomes do not affect the place draw. Defaults are `dev=50`, `seed=0`.

Outputs are `outputs/stage3/dev/cohort.jsonl` and `summary.json`. Each record
contains the original `image_key`, `place_key`, city/local-place identity,
`role`/`source_role=SOURCE`, absolute `source_path`, a `source_identity` object
with the five unchanged split identity fields, and `source_sha256` computed
from the selected image bytes at build time. No image is decoded or copied.
Missing selected images fail rather than silently changing the cohort.

The summary includes exact Stage2 exclusions and target overlap, input hashes,
the selection rule, dev membership, cohort hash and disjointness checks.
The reference definition remains **all 127,118 SUPPORT images from the frozen
split**, including SUPPORT at Stage2-100 places: same-place SUPPORT images are
positives, all other-place SUPPORT images are negatives. Query exclusions do
not filter the reference database. The embedded Stage2 retrieval identity is
recorded metadata; this builder does not recompute descriptors or rehash SUPPORT
image bytes. The split hash pins the full SUPPORT membership and ordering.

Only dev is created; remaining candidates are neither assigned to nor exported
as eval. This module does not import Stage2 runtime code, mine vulnerabilities,
load models, or invoke Diffusion. Stage1/Stage2 and existing targets are read-only.
Publication is staged; identical reruns leave files untouched, and an existing
different output is refused. Paths are resolved relative to the repository,
so the CLI also works from another working directory.

## Frozen vulnerability export for dev (schema version 1)

```bash
conda run -n boq python scripts/stage3_export_vulnerability.py
python -m pytest -q tests/test_stage3_vulnerability_export.py
```

This runs the existing `src.stage2.vulnerability.VulnerabilityEstimator` for
every frozen Stage3-dev SOURCE, using the Stage2 checkpoint and full SUPPORT
descriptor cache. It verifies cohort/source hashes, split identities, Stage2
exclusions, producer code hashes, cache bytes and retrieval identity before
export. No training, Stage2 changes, threshold overrides, query resampling,
SUPPORT filtering, or eval selection occurs. The CLI intentionally exposes no
estimator or threshold tuning flags.

The current `src/stage2/retrieval.py` has comment-only changes since the recorded
Stage2 run. For a differing producer hash, the exporter first verifies the
archived file at commit `1993f93ba7b6df1816ef46ae482c62cd2130326a` against the
recorded SHA-256, then requires identical non-comment Python tokens (including
literals/operators/indentation). Any computational edit fails. Both archival
and current hashes and this verification result are recorded; the cache retains
its original identity, and its current runtime identity is recorded separately.

Stage2 settings are unchanged: 224×224 bicubic-antialias uint8 resize then
float32 scaling; ImageNet normalization after RGB mean-fill; 32 attention-ranked
2×2 probe windows, stride 1, batch size 32, fused alpha 0.5. Diagnostic thresholds
remain those recorded in Stage2 verification. As in the Stage2 experiment,
`strict=False` retains all diagnostic STOP cases and their reasons, without
filtering the cohort. Descriptor inconsistency is a hard export failure.

Output: `outputs/stage3/dev/vulnerability/{vulnerability.jsonl,summary.json,maps/}`.
Each JSONL record preserves source/place identity and source hash, records raw
decoded original height/width (without EXIF transpose), clean retrieval metrics,
all diagnostics, per-window metrics, and the relative NPZ path/hash. Each
pickle-free NPZ repeats the image/place identity and original dimensions, with:

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `clean_descriptor` | 8192 / float32 | Unchanged normalized BoQ descriptor |
| `clean_margin` | scalar / float64 container | Frozen full-SUPPORT best-positive minus hardest-negative cosine margin |
| `raw_attention_map` | 16×16 / float32 | Primary Attention Proposal Map; mean heads → queries → layers |
| `attention_map` | 16×16 / float64 | Stage2 `normalize_scores(raw_attention_map)` |
| `intervention_damage_map` | 16×16 / float32 | Overlap-mean positive margin drop over measured windows |
| `intervention_map` | 16×16 / float64 | Stage2 normalized intervention damage |
| `fused_map` | 16×16 / float64 | 0.5 attention + 0.5 intervention |
| `attention_roi_token_mask` | 16×16 / bool | Primary connected_topk ROI, constructed exactly as in Stage2 |
| `fused_roi_token_mask` | 16×16 / bool | Supplementary connected ROI returned by the estimator |
| `coverage_counts` | 16×16 / int64 | Number of measured probe windows covering each token |
| `selected_windows` | 32×2 / int64 | Probe top-left token coordinates, in attention-rank order |
| `window_attention_scores`, `window_margin_drops` | 32 / float32 | Probe scores and signed margin drops |

Both masks contain `round(0.15 * 256) = 38` tokens (actual fraction 0.1484375).
The separate continuous maps retain their exact estimator values and dtypes;
they are never binarized, quantized, resized, or replaced by the ROI. Scientific
coverage should consume the saved token-space weights with token-aligned
regions and explicitly name the selected map. Raw attention and the Stage2
normalized maps are distinct weight definitions. No PNGs are generated here;
any later visualization or pixel interpolation must remain display-only.
Unprobed intervention tokens have zero damage and explicit zero coverage count;
window damage is a ranking signal, not a measured per-token causal effect.

For each image the adapter independently calls the standard Stage2
`evaluation_transform` + `model.forward` and compares against the extracted
descriptor, in addition to retaining the estimator's own pre/post-forward
checks. The frozen descriptor tolerance is 1e-6; a direct-forward full-SUPPORT
margin check uses the same tolerance. The summary reports all maximum errors,
the full [127118, 8192] reference shape, settings, thresholds, code/input hashes,
and diagnostic GO/STOP counts. `status=COMPLETE` means a complete validated
export, not a claim that all estimator diagnostics passed.

NPZ arrays are round-trip checked with `allow_pickle=False`, including exact
values/dtypes/identity. Publication is atomic after all 50 records pass export
checks. Existing exports are refused, so downstream numerical artifacts cannot
be silently replaced.

## Dev Core-mask mean-fill evaluation

```bash
conda run -n boq python scripts/stage3_eval_core_masks.py
python -m pytest -q tests/test_stage3_core_mask_eval.py
```

The evaluator consumes the frozen dev cohort/vulnerability export and AdaptVPR
`outputs/stage3_dev/candidate_masks`. It uses the candidate-audited subset
(currently the first 20 of 50 dev queries), never substitutes unevaluated queries,
and reports missing masks at each of the three predeclared weighted-coverage
thresholds, **0.3 / 0.5 / 0.7**, with target precision fixed at **0.7**.
Continuous weights remain in token space. Selected candidates' numerical
coverage is independently recomputed from native pixel occupancy per token;
neither visualization PNG values nor Render Masks enter coverage or retrieval.

Within each query/threshold, selection maximizes full-map weighted coverage,
then target precision, minimizes area and centroid distance, and finally breaks
ties by candidate ID. This rule and the complete selected masks are saved
**before loading BoQ**. Selection never uses retrieval outcomes. The baseline
calls the unchanged old AdaptVPR ellipse adapter, requesting 6% image area
(its original relative area tolerance is ±5%) and 70% target precision; adapter
failure is recorded. The ellipse is not subjected to family weighted gates.

The original RGB SOURCE is mean-filled within each native Core. Five random
controls are direct integer translations of that same native boolean mask,
sampled uniformly over all in-bounds placements other than the original, **with
replacement**, as in Stage2's pixel-control helper. Target overlap is allowed;
duplicate draws are retained and logged. Masks are never resized, rotated,
deformed or clipped. Every control must have exactly the target's coordinates
plus its recorded offset, preserving holes, connectivity, shape and area.
No legal translation means no paired result, with an explicit exclusion.

Mean-fill uses the original image's per-channel spatial mean, shared by all
placements. Float RGB intervention values are explicitly rounded to uint8,
then all RGB images pass through the unchanged Stage2 uint8 bicubic-antialias
224×224 resize and ImageNet normalization. The requested and applied fill values
are saved. This native-pixel intervention protocol differs from Stage2's
224-space intervention; **the model, preprocessing and retrieval margin
definition remain frozen**. Only RGB images are resized, never masks. Clean
descriptors and margins are independently compared against the vulnerability
export at tolerance 1e-6. The original checkpoint and full 127,118 SUPPORT cache
are hash/identity checked, with the same comment-only code audit described above.

Primary margin drop is `clean_margin - perturbed_margin`, where margin is the
best same-place SUPPORT cosine minus the hardest other-place SUPPORT cosine.
The five random drops are averaged **within query** before statistical analysis.
Paired advantage is `target_drop - mean(random_drop)`; positive values favor
the targeted Core. Statistics reuse the existing analysis helper: mean/median,
20,000 paired query bootstrap resamples, percentile 95% CI of the mean paired
advantage, two-sided Wilcoxon (`zero_method=wilcox`, asymptotic), and win/tie rate.
Arm-specific deterministic bootstrap streams use the declared seed. Win/tie and
Wilcoxon use 12-decimal differences, as in the existing helper. Fewer than two
pairs yields no inferential statistics; zero-pair arms remain explicit outputs.

All statistics and p-values are **dev configuration selection only**, not final
paper tests. The predeclared freeze rule considers family arms with at least two
paired queries, prioritizes number of eligible queries, then mean paired margin
advantage, then the higher threshold. It never selects by p-value. A frozen dev
configuration records the candidate rules, intervention, checkpoint/cache hashes
and seed; a tiny eligible subset is not evidence of population-level validity.
Family-versus-ellipse comparisons use only common query identities and compare
each mask's target-minus-own-random advantage, not mismatched cohort means.

Output directory: `outputs/stage3/dev/core_mask_eval/`. An existing directory is
refused; incomplete runs have no `summary.json` with `status=COMPLETE`.

- `protocol.json`, `chosen_core_masks.json`: pre-response rules, input hashes,
  selected footprint, coverage and identity; target PNGs in `masks/`.
- `per_condition.csv`: clean, target, and five random rows per eligible mask;
  `paired.csv`: one row per query/arm after averaging random repeats. The same
  CSVs also appear in each arm directory, including headers for empty arms.
- `statistics.csv`, `summary.json`: arm sample sizes and query-level statistics.
- `random_placements.json`, `masks/*_placements.npz`: native target then random
  boolean masks, exact offsets, hashes, fill values and clean-forward checks.
- `ellipse_common_queries.csv`, `ellipse_common_statistics.json`: paired
  family/ellipse comparison on their intersection.
- `exclusions.json`, `ellipse_*_diagnostic.json`: every gate/adapter failure.
- `frozen_config.json`: explicit dev freeze result; no hidden threshold choice.

The CLI exposes only input/output locations and seed. It does not run a VLM,
vulnerability mining, Diffusion, training, or modify Stage1/Stage2, their
thresholds, or the dev split. It never constructs or opens final eval.

## Existing target export (schema version 1)

Export existing Stage2 masks only. No inference, vulnerability mining, statistical
reanalysis, checkpoint loading, image copying, or changes to Stage1/Stage2.
Requires Python, NumPy and Pillow; the exporter does not import PyTorch.

```bash
python scripts/stage3_export_targets.py \
  --stage2-commit 1993f93ba7b6df1816ef46ae482c62cd2130326a \
  --mask-mode connected_topk
```

Default input: `outputs/stage2/gsv_occlusion/support`; default output:
`outputs/stage3/targets/{targets.jsonl,export_manifest.json,masks/}`. All recorded
seeds are exported. `--seeds 0` selects an existing seed, never resamples it.
`--include-fused` also exports fused with `target_role="supplementary"`;
attention always has `target_role="primary"`. Prototype artifacts can be selected
with `--stage2-dir outputs/stage2/gsv_occlusion/prototype` and a separate output
directory. Reference modes are not mixed or deduplicated silently.

## Missing legacy metadata and provenance

Existing Stage2 JSON does **not** record `stage2_commit` or `mask_mode`. Omitting
these explicit arguments fails with a missing-field error. The values above were
verified against Git: that commit contains the exact artifacts and code whose
SHA256 hashes Stage2 recorded. `stage2_commit` means the archival commit containing
the verified artifacts/code, **not a claim about the HEAD at experiment execution**.
The manifest documents this distinction and the explicit metadata supplied.

The legacy producer adapter accepts only the audited hashes in `contracts.py`:
attention uses the `connected_topk` default; fused specifies that mode explicitly.
This is not a fallback for arbitrary missing metadata. Unknown producer hashes,
missing required fields, failed Stage2 gates or inconsistent identities fail.
All consumed artifacts, including ordered cohorts, NPZ, split, diagnostics and
CSV, must byte-match the supplied commit. NPZ has no embedded image identities:
Git anchoring protects its `query_NNN_strategy` to cohort binding, including
same-area mask swaps that geometry checks alone cannot detect. As with any
artifact contract, this trusts the original producer's recorded binding.

`bag_of_queries_head` records the export-time repository HEAD. Input hashes,
producer hashes, exporter hashes and tracked working-tree diff hash are included.
A checkpoint's path and SHA256 are copied from the hash-verified retrieval report;
the checkpoint need not be present or loaded. SOURCE bytes were not hashed by
Stage2: `source_sha256` pins the file **at export time**, not retroactively at
Stage2 execution. Source dimensions come from the original image's raw decoded
orientation, matching Stage2 (no EXIF transpose).

## Consumer contract

- `image_key` is unchanged from the Stage2 split, relative to `images_root`.
  `place_key` and `source_role="SOURCE"` are checked against split, cohort,
  diagnostics and CSV. `source_path` is absolute; no source image is copied.
  On another machine, join your dataset root to `image_key` and verify its hash.
- `mask_224_path` and `mask_original_path` are relative to the export directory.
  PNGs are single-channel uint8 with values **0 and 1**, where 1 selects the
  target. Do not divide by 255. File SHA256 values are in each target record.
- `mask_ratio` is the requested token ratio; `mask_token_count` is exactly
  Python `round(mask_ratio * token_grid_height * token_grid_width)`.
- Resize uses floor-coordinate nearest neighbor, matching Stage2's PyTorch
  `interpolate(mode="nearest")`: tokens -> 224x224 -> original resolution.
  This explicitly differs from libraries' center-coordinate nearest variants.
  The exported original mask is derived from the exported 224 mask.
- `actual_pixel_area_224` / `actual_pixel_area_original` count selected pixels;
  `actual_area_fraction_224` / `actual_area_fraction_original` divide those counts
  by their respective image areas. Resizing can change the area fraction.
- `clean_margin` is the Stage2 CSV value for that SOURCE/seed/reference mode.
  `stage2_query_index` and `stage2_mask_key` identify the original NPZ member.
- `sample_id` hashes schema, input config hash, seed, image_key and target type.
  No timestamps or export-directory paths enter the output. For fixed inputs,
  exporter code and Git state, reruns are byte-identical. An existing identical
  export is accepted; a different existing export is refused. New exports are
  staged and atomically published only after all checks pass.

```bash
python -m pytest -q tests/test_stage3_export_targets.py
```
