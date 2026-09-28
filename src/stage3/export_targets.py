"""Strict, offline export from the existing Stage2 GSV occlusion artifact."""

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from .contracts import (
    ExportError, LEGACY_PRODUCER_SHA256, SCHEMA_VERSION, TARGET_ROLES,
    binary_mask, json_text, nearest_resize, require, sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FIELDS = ("image_key", "place_key", "city_id", "local_place_id", "role")


def _git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ExportError(f"Git provenance check failed: {' '.join(args)}: {result.stderr.decode().strip()}")
    return result.stdout


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _number(row, key, context, integer=False):
    require(row, [key], context)
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{context}: invalid {key}") from exc
    if not math.isfinite(value) or (integer and value != int(value)):
        raise ExportError(f"{context}: invalid {key}")
    return int(value) if integer else value


def _verify_commit(root, commit, inputs, source_hashes):
    """The commit anchors exact artifacts, including NPZ order-to-image binding."""
    commit = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    for path, expected in inputs.items():
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ExportError(f"Stage2 input must be inside the provenance repository: {path}") from exc
        actual = hashlib.sha256(_git(root, "show", f"{commit}:{relative}")).hexdigest()
        if actual != expected:
            raise ExportError(f"stage2_commit artifact mismatch: {relative}")
    for path, expected in source_hashes.items():
        if hashlib.sha256(_git(root, "show", f"{commit}:{path}")).hexdigest() != expected:
            raise ExportError(f"stage2_commit producer code mismatch: {path}")
    return commit


def _split_sources(path, wanted):
    found = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            require(row, IDENTITY_FIELDS, f"split line {line_no}")
            if row["image_key"] in wanted:
                if row["image_key"] in found:
                    raise ExportError(f"Duplicate image_key in split: {row['image_key']}")
                found[row["image_key"]] = row
    if set(found) != wanted:
        raise ExportError("Cohort image_key missing from Stage2 split")
    return found


def export_targets(*, stage2_dir, output_dir, repo_root=ROOT, stage2_commit=None,
                   mask_mode=None, seeds=None, include_fused=False):
    """Publish atomically; identical reruns are no-ops, differing exports fail.

    Legacy config lacks commit/mask_mode. Both must be supplied explicitly and
    checked against the immutable Git artifacts and audited producer hashes.
    Source dimensions are read from the original image, never guessed.
    """
    root = Path(repo_root).resolve()
    stage2_dir = _repo_path(root, stage2_dir).resolve()
    output_dir = _repo_path(root, output_dir).resolve()
    protected = [root / "outputs/stage1", root / "outputs/stage2", stage2_dir]
    if any(output_dir == p or p in output_dir.parents or output_dir in p.parents for p in protected):
        raise ExportError("Output must not overlap Stage1/Stage2 artifacts")
    config_path = stage2_dir / "config.json"
    config = _read_json(config_path)
    require(config, ["args", "cohorts", "retrieval_identity", "source_sha256",
                     "retrieval_report_sha256", "protocol"], "Stage2 config")
    for key, override in (("stage2_commit", stage2_commit), ("mask_mode", mask_mode)):
        if key not in config and override is None:
            raise ExportError(f"Stage2 config missing {key}; explicitly supply --{key.replace('_', '-')} "
                              "with verifiable provenance; no default is inferred")
        if key in config and override is not None and config[key] != override:
            raise ExportError(f"Explicit {key} disagrees with Stage2 config")
    commit = stage2_commit or config["stage2_commit"]
    mode = mask_mode or config["mask_mode"]
    if mode != "connected_topk" or any(config["source_sha256"].get(k) != v
                                        for k, v in LEGACY_PRODUCER_SHA256.items()):
        raise ExportError("Unsupported producer/mask_mode; require the audited connected_topk producer hashes")
    args, identity = config["args"], config["retrieval_identity"]
    require(args, ["seeds", "num_queries", "mask_ratio", "reference_mode", "retrieval_report", "split"], "Stage2 args")
    require(identity, ["split_sha256", "checkpoint_sha256", "images_root", "image_size", "source_sha256"], "retrieval identity")
    require(config["protocol"], ["primary", "supplementary"], "protocol")
    if config["protocol"]["primary"] != "attention" or config["protocol"]["supplementary"] != "fused":
        raise ExportError("Stage2 target roles disagree with attention primary / fused supplementary")
    if identity["image_size"] != [224, 224]:
        raise ExportError("Stage2 image_size must be [224, 224]")
    all_seeds = args["seeds"]
    if (not isinstance(all_seeds, list) or not all_seeds or
            any(type(s) is not int or s < 0 for s in all_seeds) or len(set(all_seeds)) != len(all_seeds)):
        raise ExportError("Stage2 seeds must be distinct nonnegative integers")
    selected = sorted(all_seeds if seeds is None else seeds)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(all_seeds):
        raise ExportError("Requested seeds must be distinct existing Stage2 seeds")
    ratio = _number(args, "mask_ratio", "args")
    if not 0 < ratio <= 1:
        raise ExportError("mask_ratio must be in (0, 1]")
    count = _number(args, "num_queries", "args", integer=True)
    if count < 1 or set(config["cohorts"]) != {str(s) for s in all_seeds}:
        raise ExportError("Stage2 cohorts do not match configured seeds/count")
    split_path = _repo_path(root, args["split"])
    report_path = _repo_path(root, args["retrieval_report"])
    summary_path = stage2_dir / "summary.json"
    diagnostics_path = stage2_dir / "estimator_diagnostics.json"
    csv_path = stage2_dir / "per_query.csv"
    verification_path = stage2_dir.parent / "verification.json"
    report, summary, verification = map(_read_json, (report_path, summary_path, verification_path))
    require(report, ["status", "identity", "checkpoint_path"], "retrieval report")
    require(summary, ["status", "reference_mode"], "Stage2 summary")
    reference = args["reference_mode"]
    require(verification, [reference], "Stage2 verification")
    if (report["status"] != "GO" or summary["status"] != "GO" or
            verification[reference].get("status") != "PASS" or summary["reference_mode"] != reference):
        raise ExportError("Stage2 validation must be GO/PASS for this reference_mode")
    if report["identity"] != identity:
        raise ExportError("Retrieval identity mismatch")
    if sha256_file(report_path) != config["retrieval_report_sha256"]:
        raise ExportError("Retrieval report SHA256 mismatch")
    if sha256_file(split_path) != identity["split_sha256"]:
        raise ExportError("Stage2 split SHA256 mismatch")
    paths = [config_path, split_path, report_path, summary_path, diagnostics_path, csv_path, verification_path]
    paths += [stage2_dir / f"target_masks_seed{s}.npz" for s in all_seeds]
    inputs = {p: sha256_file(p) for p in paths}

    cohorts = {}
    for seed in all_seeds:
        rows = config["cohorts"][str(seed)]
        if not isinstance(rows, list) or len(rows) != count:
            raise ExportError(f"Incomplete Stage2 cohort for seed {seed}")
        for row in rows:
            require(row, IDENTITY_FIELDS, f"cohort seed={seed}")
            key = row["image_key"]
            key_path = PurePosixPath(key)
            if (key_path.is_absolute() or ".." in key_path.parts or "\\" in key or
                    key_path.as_posix() != key or len(key_path.parts) != 2):
                raise ExportError(f"Unsafe image_key: {key}")
            if row["role"] != "SOURCE" or row["place_key"] != f"{row['city_id']}:{row['local_place_id']}":
                raise ExportError("Cohort must identify a SOURCE and a consistent place_key")
            if key_path.parts[0] != row["city_id"]:
                raise ExportError("image_key city differs from place_key")
        if len({r["image_key"] for r in rows}) != count or len({r["place_key"] for r in rows}) != count:
            raise ExportError("Duplicate SOURCE image/place in Stage2 cohort")
        cohorts[seed] = rows
    split = _split_sources(split_path, {r["image_key"] for rows in cohorts.values() for r in rows})
    for rows in cohorts.values():
        for row in rows:
            if any(row[k] != split[row["image_key"]][k] for k in IDENTITY_FIELDS):
                raise ExportError(f"Cohort SOURCE identity differs from Stage2 split: {row['image_key']}")

    expected = {(s, r["image_key"]): r for s, rows in cohorts.items() for r in rows}
    diagnostics = {}
    for row in _read_json(diagnostics_path):
        require(row, ["seed", "image_key", "place_key", "token_grid", "image_size", "mask_ratio", "mask_tokens"], "diagnostic")
        key = (row["seed"], row["image_key"])
        if key in diagnostics or key not in expected or row["place_key"] != expected[key]["place_key"]:
            raise ExportError("Diagnostic SOURCE identity mismatch or duplicate")
        diagnostics[key] = row
    if set(diagnostics) != set(expected):
        raise ExportError("Missing Stage2 SOURCE diagnostics")
    metrics = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            require(row, ["seed", "image_key", "place_key", "strategy", "condition"], "per_query.csv")
            key = (_number(row, "seed", "CSV", integer=True), row["image_key"])
            if key not in expected or row["place_key"] != expected[key]["place_key"]:
                raise ExportError("CSV SOURCE identity mismatch")
            if row["strategy"] not in TARGET_ROLES or row["condition"] not in ("clean", "targeted", "random"):
                raise ExportError("Unknown Stage2 strategy/condition")
            if row["condition"] in ("clean", "targeted"):
                metric_key = (*key, row["strategy"], row["condition"])
                if metric_key in metrics:
                    raise ExportError("Duplicate clean/targeted row")
                metrics[metric_key] = row
    required_metrics = {(*k, t, c) for k in expected for t in TARGET_ROLES for c in ("clean", "targeted")}
    if set(metrics) != required_metrics:
        raise ExportError("Missing clean/targeted Stage2 metrics")
    source_hashes = dict(identity["source_sha256"])
    for path, digest in config["source_sha256"].items():
        if path in source_hashes and source_hashes[path] != digest:
            raise ExportError(f"Conflicting producer code identity: {path}")
        source_hashes[path] = digest
    commit = _verify_commit(root, commit, inputs, source_hashes)
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    image_root = _repo_path(root, identity["images_root"]).resolve()
    if output_dir == image_root or image_root in output_dir.parents or output_dir in image_root.parents:
        raise ExportError("Output must not overlap SOURCE images")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".stage3-export-", dir=output_dir.parent))
    try:
        (temporary / "masks").mkdir()
        records = []
        for seed in selected:
            with np.load(stage2_dir / f"target_masks_seed{seed}.npz", allow_pickle=False) as archive:
                names = {f"query_{i:03d}_{t}" for i in range(count) for t in TARGET_ROLES}
                if len(archive.files) != len(names) or set(archive.files) != names:
                    raise ExportError(f"NPZ keys do not match cohort for seed {seed}")
                for index, source in enumerate(cohorts[seed]):
                    key = (seed, source["image_key"])
                    diagnostic = diagnostics[key]
                    source_path = image_root / source["image_key"]
                    if not source_path.resolve().is_relative_to(image_root):
                        raise ExportError("SOURCE path escapes images_root")
                    with Image.open(source_path) as im:
                        width, height = im.size  # raw decoded orientation, as in Stage2; no EXIF transpose
                        im.verify()
                    source_digest = sha256_file(source_path)
                    for target_type in TARGET_ROLES:
                        mask_key = f"query_{index:03d}_{target_type}"
                        token = binary_mask(archive[mask_key], mask_key)
                        budget = round(ratio * token.size)
                        if budget <= 0 or int(token.sum()) != budget:
                            raise ExportError(f"{mask_key}: token budget mismatch")
                        if (list(token.shape) != diagnostic["token_grid"] or diagnostic["image_size"] != [224, 224]
                                or diagnostic["mask_ratio"] != ratio or diagnostic["mask_tokens"] != budget):
                            raise ExportError(f"{mask_key}: diagnostic geometry/budget mismatch")
                        mask224 = nearest_resize(token, 224, 224)
                        area224 = int(mask224.sum())
                        clean = metrics[(*key, target_type, "clean")]
                        target = metrics[(*key, target_type, "targeted")]
                        margin = _number(target, "clean_margin", mask_key)
                        for metric in (clean, target):
                            if (_number(metric, "target_mask_tokens", mask_key, True) != budget or
                                    _number(metric, "mask_pixels", mask_key, True) != area224 or
                                    _number(metric, "actual_area_fraction", mask_key) != area224 / 224**2 or
                                    _number(metric, "clean_margin", mask_key) != margin):
                                raise ExportError(f"{mask_key}: CSV area/budget/margin mismatch")
                        if (_number(clean, "applied_mask_pixels", mask_key, True) != 0 or
                                _number(target, "applied_mask_pixels", mask_key, True) != area224):
                            raise ExportError(f"{mask_key}: applied pixel area mismatch")
                        if target_type == "fused" and not include_fused:
                            continue
                        original = nearest_resize(mask224, height, width)
                        area_original = int(original.sum())
                        if not area_original:
                            raise ExportError(f"{mask_key}: nearest resize erased the target")
                        sample_id = hashlib.sha256(json_text([
                            SCHEMA_VERSION, inputs[config_path], seed, source["image_key"], target_type
                        ]).encode()).hexdigest()
                        mask224_path = f"masks/{sample_id}_224.png"
                        original_path = f"masks/{sample_id}_original.png"
                        Image.fromarray(mask224).save(temporary / mask224_path)
                        Image.fromarray(original).save(temporary / original_path)
                        records.append(dict(
                            schema_version=SCHEMA_VERSION, sample_id=sample_id,
                            image_key=source["image_key"], place_key=source["place_key"], source_role="SOURCE",
                            source_path=str(source_path), source_width=width, source_height=height,
                            source_sha256=source_digest, target_type=target_type, target_role=TARGET_ROLES[target_type],
                            reference_mode=reference, mask_ratio=ratio, mask_mode=mode, mask_token_count=budget,
                            token_grid=list(token.shape), mask_224_path=mask224_path, mask_original_path=original_path,
                            actual_pixel_area_224=area224, actual_pixel_area_original=area_original,
                            actual_area_fraction_224=area224 / 224**2,
                            actual_area_fraction_original=area_original / (width * height),
                            clean_margin=margin, checkpoint_path=report["checkpoint_path"],
                            checkpoint_sha256=identity["checkpoint_sha256"], stage2_commit=commit,
                            bag_of_queries_head=head, seed=seed, stage2_query_index=index, stage2_mask_key=mask_key,
                            mask_224_sha256=sha256_file(temporary / mask224_path),
                            mask_original_sha256=sha256_file(temporary / original_path),
                        ))
        if len({r["sample_id"] for r in records}) != len(records):
            raise ExportError("Duplicate exported sample_id")
        (temporary / "targets.jsonl").write_text("".join(map(json_text, records)), encoding="utf-8")
        exporter_files = [Path(__file__), Path(__file__).with_name("contracts.py"),
                          ROOT / "scripts/stage3_export_targets.py"]
        manifest = dict(
            schema_version=SCHEMA_VERSION, record_count=len(records), seeds=selected,
            target_roles={t: role for t, role in TARGET_ROLES.items() if t == "attention" or include_fused},
            reference_mode=reference, stage2_commit=commit, bag_of_queries_head=head,
            stage2_commit_semantics="commit containing byte-identical Stage2 artifacts and producer code; not an asserted execution-time HEAD",
            explicit_legacy_metadata={"stage2_commit": stage2_commit, "mask_mode": mask_mode},
            producer_source_sha256=source_hashes,
            artifact_inputs={p.resolve().relative_to(root).as_posix(): h for p, h in inputs.items()},
            targets_sha256=sha256_file(temporary / "targets.jsonl"),
            exporter_source_sha256={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in exporter_files},
            tracked_worktree_diff_sha256=hashlib.sha256(_git(root, "diff", "HEAD", "--binary")).hexdigest(),
            mask_encoding="PNG uint8 values {0,1}; 1 = selected target",
            mask_resize="nearest floor-coordinate: token -> 224x224 -> original (raw decoded width,height)",
            path_contract="mask paths relative to this manifest; source_path absolute; image_key relative to images_root",
            images_root=str(image_root), source_files_copied=False,
            source_identity_semantics="Stage2 split image_key; source_sha256 measured at export (Stage2 did not hash SOURCE bytes)",
        )
        (temporary / "export_manifest.json").write_text(json_text(manifest), encoding="utf-8")
        # Re-check input files before publishing, to catch concurrent changes.
        if any(sha256_file(p) != h for p, h in inputs.items()):
            raise ExportError("Stage2 inputs changed during export")
        if output_dir.exists():
            old = {p.relative_to(output_dir): sha256_file(p) for p in output_dir.rglob("*") if p.is_file()}
            new = {p.relative_to(temporary): sha256_file(p) for p in temporary.rglob("*") if p.is_file()}
            if old != new:
                raise ExportError("Output already exists with different content; choose a new --output-dir")
        else:
            temporary.rename(output_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
