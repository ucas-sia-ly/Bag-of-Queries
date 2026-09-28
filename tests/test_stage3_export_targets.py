"""Exercise export with real files and immutable Git provenance; no model calls."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from src.stage3.contracts import ExportError, LEGACY_PRODUCER_SHA256, json_text, nearest_resize, sha256_file
from src.stage3.export_targets import ROOT, export_targets


def write_json(path, value):
    path.write_text(json_text(value), encoding="utf-8")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    stage2 = root / "outputs/stage2/gsv_occlusion/support"
    stage2.mkdir(parents=True)
    images = root / "images/City"
    images.mkdir(parents=True)
    records = [dict(image_key=f"City/{i}.png", place_key=f"City:{i}", city_id="City", local_place_id=i,
                    role="SOURCE") for i in range(2)]
    for i, size in enumerate([(641, 479), (319, 517)]):
        Image.new("RGB", size, (i * 80, 50, 100)).save(images / f"{i}.png")
    split_path = root / "split.jsonl"
    split_path.write_text("".join(map(json_text, records)), encoding="utf-8")
    identity = dict(split_sha256=sha256_file(split_path), checkpoint_sha256="a" * 64,
                    images_root=str(images.parent), image_size=[224, 224], source_sha256={})
    report_path = root / "retrieval.json"
    write_json(report_path, dict(status="GO", identity=identity, checkpoint_path="/recorded/model.ckpt"))
    for name, expected in LEGACY_PRODUCER_SHA256.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        assert sha256_file(target) == expected
    config = dict(args=dict(seeds=[0, 1], num_queries=2, mask_ratio=.15, reference_mode="support",
                            retrieval_report=str(report_path), split=str(split_path)),
                  cohorts={"0": records, "1": records[::-1]}, retrieval_identity=identity,
                  source_sha256=LEGACY_PRODUCER_SHA256, retrieval_report_sha256=sha256_file(report_path),
                  protocol=dict(primary="attention", supplementary="fused"))
    write_json(stage2 / "config.json", config)
    write_json(stage2 / "summary.json", dict(status="GO", reference_mode="support"))
    write_json(stage2.parent / "verification.json", dict(support=dict(status="PASS")))
    diagnostics, metrics = [], []
    for seed in (0, 1):
        masks = {}
        for index, row in enumerate(config["cohorts"][str(seed)]):
            diagnostics.append(dict(seed=seed, image_key=row["image_key"], place_key=row["place_key"],
                                    token_grid=[16, 16], image_size=[224, 224], mask_ratio=.15, mask_tokens=38))
            for strategy in ("attention", "fused"):
                token = np.zeros((16, 16), dtype=bool)
                token.flat[16 * index:16 * index + 38] = True
                if strategy == "fused":
                    token = np.flip(token, axis=1)
                masks[f"query_{index:03d}_{strategy}"] = token
                for condition in ("clean", "targeted"):
                    metrics.append(dict(seed=seed, image_key=row["image_key"], place_key=row["place_key"],
                                        strategy=strategy, condition=condition, target_mask_tokens=38,
                                        mask_pixels=7448, actual_area_fraction=7448 / 224**2,
                                        clean_margin=.25 + index / 10, applied_mask_pixels=0 if condition == "clean" else 7448))
        # Deliberately reverse NPZ physical order; the exporter must join by key.
        np.savez_compressed(stage2 / f"target_masks_seed{seed}.npz", **dict(reversed(list(masks.items()))))
    write_json(stage2 / "estimator_diagnostics.json", diagnostics)
    with (stage2 / "per_query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics[::-1])
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Stage2 fixture")
    return dict(repo_root=root, stage2_dir=stage2, output_dir=root / "outputs/stage3/targets",
                stage2_commit=git(root, "rev-parse", "HEAD"), mask_mode="connected_topk")


def records(case):
    return [json.loads(line) for line in (case["output_dir"] / "targets.jsonl").read_text().splitlines()]


def fingerprints(path):
    return {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*") if p.is_file()}


def test_binary_budget_original_dimensions_and_identity(case):
    manifest = export_targets(**case, include_fused=True)
    assert manifest["record_count"] == 8
    config = json.loads((case["stage2_dir"] / "config.json").read_text())
    for row in records(case):
        source = config["cohorts"][str(row["seed"])][row["stage2_query_index"]]
        assert (row["image_key"], row["place_key"]) == (source["image_key"], source["place_key"])
        with Image.open(row["source_path"]) as im:
            assert im.size == (row["source_width"], row["source_height"])
        a = np.array(Image.open(case["output_dir"] / row["mask_224_path"]))
        b = np.array(Image.open(case["output_dir"] / row["mask_original_path"]))
        assert set(np.unique(a)) == {0, 1} and set(np.unique(b)) == {0, 1}
        assert a.shape == (224, 224)
        assert b.shape == (row["source_height"], row["source_width"])
        assert a.sum() == row["mask_token_count"] * 14**2 == row["actual_pixel_area_224"] == 7448
        assert 0 < b.sum() == row["actual_pixel_area_original"]
        # Independent oracle: each destination samples floor(y*224/H), floor(x*224/W).
        yy, xx = np.indices(b.shape)
        assert np.array_equal(b, a[yy * 224 // b.shape[0], xx * 224 // b.shape[1]])
        with np.load(case["stage2_dir"] / f"target_masks_seed{row['seed']}.npz") as z:
            assert np.array_equal(a, np.repeat(np.repeat(z[row["stage2_mask_key"]], 14, 0), 14, 1))
        assert row["target_role"] == ("primary" if row["target_type"] == "attention" else "supplementary")
        assert row["stage2_commit"] == row["bag_of_queries_head"] == case["stage2_commit"]
    assert not list(case["output_dir"].rglob("*.jpg"))
    assert manifest["source_files_copied"] is False


def test_same_seed_rerun_byte_identical_and_read_only(case):
    before = fingerprints(case["stage2_dir"])
    first = export_targets(**case, seeds=[1])
    output = fingerprints(case["output_dir"])
    assert export_targets(**case, seeds=[1]) == first
    assert fingerprints(case["output_dir"]) == output
    other = {**case, "output_dir": case["output_dir"].with_name("again")}
    export_targets(**other, seeds=[1])
    assert fingerprints(other["output_dir"]) == output
    assert fingerprints(case["stage2_dir"]) == before
    assert {r["target_type"] for r in records(case)} == {"attention"}
    assert {r["seed"] for r in records(case)} == {1}


@pytest.mark.parametrize("field", ["stage2_commit", "mask_mode"])
def test_missing_legacy_metadata_is_explicit_error(case, field):
    case[field] = None
    with pytest.raises(ExportError, match=f"missing {field}"):
        export_targets(**case)
    assert not case["output_dir"].exists()


def mutate_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    write_json(path, data)


@pytest.mark.parametrize("change,match", [
    (lambda c: c["cohorts"]["0"][0].update(place_key="City:99"), "place_key"),
    (lambda c: c["cohorts"]["0"][0].update(role="SUPPORT"), "SOURCE"),
    (lambda c: c["cohorts"]["0"][0].update(image_key="City/../wrong.png"), "Unsafe image_key"),
    (lambda c: c["cohorts"]["0"][0].pop("place_key"), "missing required fields"),
    (lambda c: c["cohorts"]["0"].reverse(), "artifact mismatch"),
    (lambda c: c["args"].pop("mask_ratio"), "missing required fields"),
])
def test_corrupt_cohort_or_missing_fields_rejected(case, change, match):
    mutate_json(case["stage2_dir"] / "config.json", change)
    with pytest.raises(ExportError, match=match):
        export_targets(**case)
    assert not case["output_dir"].exists()


def test_csv_wrong_source_rejected(case):
    path = case["stage2_dir"] / "per_query.csv"
    path.write_text(path.read_text().replace("City:0", "City:99"))
    with pytest.raises(ExportError, match="CSV SOURCE identity mismatch"):
        export_targets(**case)


def test_split_change_rejected(case):
    path = case["repo_root"] / "split.jsonl"
    path.write_text(path.read_text().replace('"SOURCE"', '"SUPPORT"'))
    with pytest.raises(ExportError, match="split SHA256"):
        export_targets(**case)


def recommit(case):
    git(case["repo_root"], "add", ".")
    git(case["repo_root"], "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "bad fixture")
    case["stage2_commit"] = git(case["repo_root"], "rev-parse", "HEAD")


@pytest.mark.parametrize("kind,match", [("binary", "binary"), ("budget", "token budget"), ("missing", "NPZ keys")])
def test_bad_mask_rejected_even_in_committed_artifact(case, kind, match):
    path = case["stage2_dir"] / "target_masks_seed0.npz"
    with np.load(path) as z:
        masks = {k: z[k].astype(float) for k in z.files}
    if kind == "binary":
        masks["query_000_attention"][0, 0] = .5
    elif kind == "budget":
        masks["query_000_attention"][0, 0] = 0
    else:
        masks.pop("query_000_attention")
    np.savez_compressed(path, **masks)
    recommit(case)
    with pytest.raises(ExportError, match=match):
        export_targets(**case)
    assert not case["output_dir"].exists()
    assert not list(case["output_dir"].parent.glob(".stage3-export-*"))


def test_equal_area_masks_swapped_between_sources_rejected(case):
    path = case["stage2_dir"] / "target_masks_seed0.npz"
    with np.load(path) as z:
        masks = {k: z[k] for k in z.files}
    a, b = "query_000_attention", "query_001_attention"
    masks[a], masks[b] = masks[b], masks[a]
    np.savez_compressed(path, **masks)
    with pytest.raises(ExportError, match="artifact mismatch"):
        export_targets(**case)


def test_missing_metric_rejected(case):
    path = case["stage2_dir"] / "per_query.csv"
    rows = list(csv.DictReader(path.open()))
    del rows[0]["clean_margin"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[1]))
        writer.writeheader()
        writer.writerows(rows)
    recommit(case)
    with pytest.raises(ExportError, match="missing required fields: clean_margin"):
        export_targets(**case)


def test_different_existing_output_never_overwritten(case):
    export_targets(**case)
    before = fingerprints(case["output_dir"])
    with pytest.raises(ExportError, match="different content"):
        export_targets(**case, include_fused=True)
    assert fingerprints(case["output_dir"]) == before


def test_unknown_seed_and_protected_output(case):
    with pytest.raises(ExportError, match="existing Stage2 seeds"):
        export_targets(**case, seeds=[99])
    with pytest.raises(ExportError, match="overlap Stage1/Stage2"):
        export_targets(**{**case, "output_dir": case["stage2_dir"]})


def test_cli_does_not_import_models_or_mining():
    code = "from scripts.stage3_export_targets import main; import sys; main(['--help'])"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and "--include-fused" in result.stdout
    code = "import scripts.stage3_export_targets; import sys; assert 'torch' not in sys.modules; assert 'src.stage2.vulnerability' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_nearest_floor_rule_noninteger_resize():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert nearest_resize(mask, 3, 3).tolist() == [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
