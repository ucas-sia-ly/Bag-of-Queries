# Stage3 target export (schema version 1)

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
