"""Offline cohort tests with synthetic metadata and bytes; no model dependencies."""

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from src.stage3.cohort import CohortError, ROOT, assert_place_disjoint, build_cohort, select_dev


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def case(tmp_path):
    rows = []
    for city in ("A", "B"):
        for place in range(5):
            for index, role in enumerate(("SUPPORT", "SUPPORT", "SOURCE", "SOURCE", "SOURCE")):
                row = dict(image_key=f"{city}/{place}_{index}.jpg", place_key=f"{city}:{place}",
                           city_id=city, local_place_id=place, role=role)
                rows.append(row)
                if role == "SOURCE":
                    image = tmp_path / "images" / row["image_key"]
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(row["image_key"].encode())
    split = tmp_path / "outputs/stage2/split/gsv_split.jsonl"
    write_jsonl(split, rows)
    sources = [row for row in rows if row["role"] == "SOURCE"]
    identity = dict(images_root=str(tmp_path / "images"), num_support=20,
                    split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    cohorts = {"0": [sources[0]], "1": [sources[3]]}
    for mode in ("support", "prototype"):
        write_json(tmp_path / f"outputs/stage2/gsv_occlusion/{mode}/config.json",
                   dict(args=dict(seeds=[0, 1], num_queries=1, reference_mode=mode),
                        retrieval_identity=identity, cohorts=cohorts))
    # Only seed0 was exported: seed1 must still be excluded in its entirety.
    write_jsonl(tmp_path / "outputs/stage3/targets/targets.jsonl",
                [dict(cohorts["0"][0], source_role="SOURCE")])
    return dict(repo_root=tmp_path, dev=4, seed=0)


def test_build_audits_all_stage2_places_and_preserves_full_support(case):
    root = case["repo_root"]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    summary = build_cohort(**case)
    rows = read_jsonl(root / "outputs/stage3/dev/cohort.jsonl")
    assert len(rows) == len({r["place_key"] for r in rows}) == 4
    assert {r["place_key"] for r in rows}.isdisjoint({"A:0", "A:1"})
    assert summary["stage2_audit"]["excluded_place_keys"] == ["A:0", "A:1"]
    assert summary["targets_audit"]["stage2_overlap_place_keys"] == ["A:0"]
    assert summary["eligible_source_place_count"] == 8
    assert summary["reference_definition"]["num_images"] == 20  # Includes excluded Stage2 places.
    assert summary["reference_definition"]["num_places"] == 10
    assert summary["reference_definition"]["cohort_filter_applied"] is False
    assert all(summary["assertions"].values())
    assert summary["eval_created"] is False
    assert not (root / "outputs/stage3/eval").exists()
    for row in rows:
        assert row["role"] == row["source_role"] == "SOURCE"
        assert row["source_identity"]["image_key"] == row["image_key"]
        assert row["source_identity"]["place_key"] == row["place_key"]
        assert Path(row["source_path"]) == root / "images" / row["image_key"]
        assert row["source_sha256"] == hashlib.sha256(Path(row["source_path"]).read_bytes()).hexdigest()
    assert all(p.read_bytes() == content for p, content in before.items())


def test_byte_identical_rerun_and_refuse_reselection(case):
    build_cohort(**case)
    output = case["repo_root"] / "outputs/stage3/dev"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    build_cohort(**case)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    with pytest.raises(CohortError, match="refusing to overwrite"):
        build_cohort(**dict(case, seed=99))
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())


def test_place_first_selection_order_independence_and_fixed_hash(case):
    sources = [r for r in read_jsonl(case["repo_root"] / "outputs/stage2/split/gsv_split.jsonl")
               if r["role"] == "SOURCE"]
    excluded = {"A:0", "A:1"}
    expected = select_dev(sources, excluded, dev=4, seed=7)
    random.Random(123).shuffle(sources)
    assert select_dev(sources, excluded, dev=4, seed=7) == expected
    assert select_dev(sources, excluded, dev=4, seed=8) != expected
    # A view-rich place must not receive extra chances in the place draw.
    extra = dict(sources[0], image_key=f"{sources[0]['city_id']}/extra.jpg")
    more = select_dev(sources + [extra], excluded, dev=4, seed=7)
    assert [r["place_key"] for r in more] == [r["place_key"] for r in expected]
    eligible = {r["place_key"] for r in sources} - excluded
    ranked = sorted(eligible, key=lambda key: (hashlib.sha256(
        f"stage3-dev-v1:place:7\0{key}".encode()).digest(), key))
    assert [r["place_key"] for r in expected] == ranked[:4]
    # Same local ID in another city remains a different place.
    all_dev = select_dev(sources, excluded, dev=8)
    assert "B:0" in {r["place_key"] for r in all_dev}


def test_identical_rerun_preserves_downstream_vulnerability(case):
    build_cohort(**case)
    downstream = case["repo_root"] / "outputs/stage3/dev/vulnerability"
    downstream.mkdir()
    artifact = downstream / "summary.json"
    artifact.write_text('{"status":"COMPLETE"}\n')
    before = artifact.read_bytes(), artifact.stat().st_mtime_ns
    build_cohort(**case)
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change, message", [
    ({"dev": 9}, "eligible SOURCE places"), ({"dev": 0}, "dev must be positive"),
    ({"seed": -1}, "seed must be"),
    ({"output_dir": "outputs/stage2/new"}, "must not overlap"),
    ({"output_dir": "outputs/stage3/targets/new"}, "must not overlap"),
    ({"output_dir": "outputs/stage3/eval"}, "never eval"),
])
def test_invalid_requests_fail_without_dev_publication(case, change, message):
    with pytest.raises(CohortError, match=message):
        build_cohort(**dict(case, **change))
    assert not (case["repo_root"] / "outputs/stage3/dev").exists()


@pytest.mark.parametrize("damage", ["missing_seed", "wrong_place", "support_source", "split_hash"])
def test_inconsistent_stage2_cohort_fails_closed(case, damage):
    path = case["repo_root"] / "outputs/stage2/gsv_occlusion/support/config.json"
    config = json.loads(path.read_text())
    if damage == "missing_seed":
        del config["cohorts"]["1"]
    elif damage == "wrong_place":
        config["cohorts"]["0"][0].update(place_key="A:4", local_place_id=4)
    elif damage == "support_source":
        config["cohorts"]["0"][0]["image_key"] = "A/0_0.jpg"
    else:
        config["retrieval_identity"]["split_sha256"] = "0" * 64
    write_json(path, config)
    with pytest.raises(CohortError):
        build_cohort(**case)
    assert not (case["repo_root"] / "outputs/stage3/dev").exists()


def test_unknown_target_exposure_fails(case):
    root = case["repo_root"]
    source = next(r for r in read_jsonl(root / "outputs/stage2/split/gsv_split.jsonl")
                  if r["place_key"] == "B:0" and r["role"] == "SOURCE")
    write_jsonl(root / "outputs/stage3/targets/targets.jsonl", [dict(source, source_role="SOURCE")])
    with pytest.raises(CohortError, match="outside Stage2-100"):
        build_cohort(**case)


@pytest.mark.parametrize("damage", ["duplicate", "missing_support", "unsafe_path"])
def test_invalid_split_fails(case, damage):
    root = case["repo_root"]
    path = root / "outputs/stage2/split/gsv_split.jsonl"
    rows = read_jsonl(path)
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "missing_support":
        rows.pop(0)
    else:
        rows[0]["image_key"] = "../outside.jpg"
    write_jsonl(path, rows)
    with pytest.raises(CohortError):
        build_cohort(**case)


def test_missing_selected_source_fails_without_resampling(case):
    root = case["repo_root"]
    sources = [r for r in read_jsonl(root / "outputs/stage2/split/gsv_split.jsonl") if r["role"] == "SOURCE"]
    selected = select_dev(sources, {"A:0", "A:1"}, dev=case["dev"])
    (root / "images" / selected[0]["image_key"]).unlink()
    with pytest.raises(FileNotFoundError):
        build_cohort(**case)
    assert not (root / "outputs/stage3/dev").exists()


def test_explicit_disjoint_assertions():
    row = dict(image_key="A/source.jpg", place_key="A:1", role="SOURCE")
    with pytest.raises(CohortError, match="Stage2-100"):
        assert_place_disjoint([row], {"A:1"}, set())
    with pytest.raises(CohortError, match="existing Stage3"):
        assert_place_disjoint([row], set(), {"A:1"})
    with pytest.raises(CohortError, match="one SOURCE per place"):
        assert_place_disjoint([row, dict(row, image_key="A/other.jpg")], set(), set())


def test_cli_builds_without_site_packages_or_stage2_runtime(case):
    root = case["repo_root"]
    command = [sys.executable, "-S", str(ROOT / "scripts/stage3_build_cohort.py"),
               "--split", str(root / "outputs/stage2/split/gsv_split.jsonl"),
               "--stage2-configs", str(root / "outputs/stage2/gsv_occlusion/support/config.json"),
               str(root / "outputs/stage2/gsv_occlusion/prototype/config.json"),
               "--targets", str(root / "outputs/stage3/targets/targets.jsonl"),
               "--output-dir", str(root / "outputs/stage3/dev"), "--dev", "4"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dev"] == 4
