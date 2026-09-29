"""Independent, standard-library-only Stage3 dev selection from a frozen split.

Only selected dev image bytes are read. No eval cohort, image decoding, model
imports, retrieval, mining, or reference subset is created.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FIELDS = ("image_key", "place_key", "city_id", "local_place_id", "role")
SELECTION_VERSION = "stage3-dev-v1"


class CohortError(ValueError):
    """Invalid inputs or a cohort that would violate the frozen protocol."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")) + "\n"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except ValueError as exc:
                raise CohortError(f"{path}:{number}: {exc}") from exc
            yield row


def _identity(row):
    try:
        identity = {key: row[key] for key in IDENTITY_FIELDS}
        key = PurePosixPath(identity["image_key"])
        if (key.is_absolute() or len(key.parts) != 2 or ".." in key.parts
                or "\\" in identity["image_key"] or key.as_posix() != identity["image_key"]
                or key.parts[0] != identity["city_id"]):
            raise ValueError("unsafe image_key or city mismatch")
        if (type(identity["local_place_id"]) is not int or identity["local_place_id"] < 0
                or identity["place_key"] != f"{identity['city_id']}:{identity['local_place_id']}"
                or identity["role"] not in ("SOURCE", "SUPPORT")):
            raise ValueError("invalid place identity or role")
        return identity
    except (KeyError, TypeError, ValueError) as exc:
        raise CohortError(f"Invalid SOURCE/SUPPORT identity: {exc}") from exc


def _rank(seed, kind, key):
    # Domain separation prevents accidental reuse of Stage2's sampling order.
    return hashlib.sha256(f"{SELECTION_VERSION}:{kind}:{seed}\0{key}".encode("utf-8")).digest(), key


def assert_place_disjoint(records, stage2_places, target_places):
    """Runtime assertions also remain active under python -O."""
    places = [row["place_key"] for row in records]
    if len(set(places)) != len(places):
        raise CohortError("Dev must contain exactly one SOURCE per place")
    if len({row["image_key"] for row in records}) != len(records):
        raise CohortError("Duplicate dev image_key")
    if set(places) & set(stage2_places):
        raise CohortError("Dev overlaps Stage2-100 places")
    if set(places) & set(target_places):
        raise CohortError("Dev overlaps existing Stage3 target places")
    if any(row["role"] != "SOURCE" for row in records):
        raise CohortError("Dev may only contain SOURCE images")


def select_dev(sources, excluded_places, *, dev=50, seed=0):
    """Hash places first, then one image per selected place; ignore input order."""
    if type(dev) is not int or dev < 1 or type(seed) is not int or seed < 0:
        raise CohortError("dev must be positive and seed must be a nonnegative integer")
    groups = defaultdict(list)
    seen = set()
    for raw in sources:
        row = _identity(raw)
        if row["role"] != "SOURCE" or row["image_key"] in seen:
            raise CohortError("Selection requires unique SOURCE images")
        seen.add(row["image_key"])
        if row["place_key"] not in excluded_places:
            groups[row["place_key"]].append(row)
    if dev > len(groups):
        raise CohortError(f"Requested dev={dev}, but only {len(groups)} eligible SOURCE places")
    places = sorted(groups, key=lambda key: _rank(seed, "place", key))[:dev]
    selected = [min(groups[key], key=lambda row: _rank(seed, "image", row["image_key"]))
                for key in places]
    assert_place_disjoint(selected, excluded_places, set())
    return selected


def build_cohort(*, repo_root=ROOT, split="outputs/stage2/split/gsv_split.jsonl",
                 stage2_configs=None, targets="outputs/stage3/targets/targets.jsonl",
                 output_dir="outputs/stage3/dev", dev=50, seed=0):
    """Audit legacy membership, select dev, and publish byte-stable artifacts.

The Stage2 exclusion set is the union of every seed0/seed1 cohort in both
reference modes, never just the images that happen to have exported targets.
Existing targets must resolve to that historical set; unknown exposure fails.
"""
    root = Path(repo_root).resolve()

    def resolve(value):
        return (root / value).resolve()

    split_path, targets_path, destination = map(resolve, (split, targets, output_dir))
    if stage2_configs is None:
        stage2_configs = [f"outputs/stage2/gsv_occlusion/{mode}/config.json"
                          for mode in ("support", "prototype")]
    configs = sorted({resolve(path) for path in stage2_configs})
    if not configs:
        raise CohortError("Stage2 seed0/seed1 configs are required")
    protected = [root / p for p in ("src/stage1", "src/stage2", "outputs/stage1", "outputs/stage2")]
    protected += [split_path, targets_path.parent, *configs]
    if any(destination == p or p in destination.parents or destination in p.parents for p in protected):
        raise CohortError("Output must not overlap Stage1/Stage2 or existing target inputs")
    output_parts = destination.relative_to(root).parts if destination.is_relative_to(root) else destination.parts
    if "eval" in output_parts:
        raise CohortError("This builder only creates dev, never eval")

    input_hashes = {str(p): _sha256(p) for p in [split_path, targets_path, *configs]}
    split_hash = input_hashes[str(split_path)]
    by_image, sources, support = {}, [], []
    counts = defaultdict(Counter)
    for raw in _jsonl(split_path):
        row = _identity(raw)
        if row["image_key"] in by_image:
            raise CohortError(f"Duplicate split image_key: {row['image_key']}")
        by_image[row["image_key"]] = row
        counts[row["place_key"]][row["role"]] += 1
        (sources if row["role"] == "SOURCE" else support).append(row)
    if len(counts) < 2 or any(c["SUPPORT"] != 2 or c["SOURCE"] < 1 for c in counts.values()):
        raise CohortError("Frozen split requires >=2 places, each with 2 SUPPORT and >=1 SOURCE")

    stage2_places, seed_places, cohort_audits = set(), defaultdict(set), []
    retrieval_identity = None
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        args, cohorts, identity = config["args"], config["cohorts"], config["retrieval_identity"]
        if (sorted(args["seeds"]) != [0, 1] or set(cohorts) != {"0", "1"}
                or type(args["num_queries"]) is not int or args["num_queries"] < 1):
            raise CohortError("Expected complete Stage2 seed0/seed1 cohorts")
        if identity["split_sha256"] != split_hash or identity["num_support"] != len(support):
            raise CohortError("Stage2 retrieval identity disagrees with frozen split")
        if retrieval_identity is not None and identity != retrieval_identity:
            raise CohortError("Stage2 configs disagree on full SUPPORT retrieval identity")
        retrieval_identity = identity
        for stage2_seed in (0, 1):
            rows = [_identity(row) for row in cohorts[str(stage2_seed)]]
            if (len(rows) != args["num_queries"] or len({r["place_key"] for r in rows}) != len(rows)
                    or any(r["role"] != "SOURCE" or by_image.get(r["image_key"]) != r for r in rows)):
                raise CohortError("Stage2 cohort has missing, duplicate or inconsistent SOURCE identities")
            places = {r["place_key"] for r in rows}
            stage2_places.update(places)
            seed_places[stage2_seed].update(places)
            cohort_audits.append(dict(config=str(path), reference_mode=args["reference_mode"],
                                     seed=stage2_seed, place_count=len(places), place_keys=sorted(places)))

    target_places, target_count = set(), 0
    for row in _jsonl(targets_path):
        original = by_image.get(row.get("image_key"))
        if (original is None or original["role"] != "SOURCE" or row.get("source_role") != "SOURCE"
                or row.get("place_key") != original["place_key"]):
            raise CohortError("Existing target SOURCE identity differs from frozen split")
        target_places.add(row["place_key"])
        target_count += 1
    if not target_places <= stage2_places:
        raise CohortError("Existing targets include places outside Stage2-100; exposure audit required")

    selected = select_dev(sources, stage2_places, dev=dev, seed=seed)
    assert_place_disjoint(selected, stage2_places, target_places)
    images_root = resolve(retrieval_identity["images_root"])
    records = []
    for row in selected:
        source_path = (images_root / row["image_key"]).resolve()
        if not source_path.is_relative_to(images_root):
            raise CohortError("SOURCE path escapes images_root")
        # Missing images fail, never change the selected cohort by resampling.
        records.append(dict(row, schema_version=1, cohort="dev", selection_seed=seed,
                            source_role="SOURCE", source_identity=dict(row),
                            source_path=str(source_path), source_sha256=_sha256(source_path)))
    cohort_text = "".join(map(_json, records))
    summary = dict(
        schema_version=1, cohort="dev", dev=dev, seed=seed, eval_created=False,
        selection=dict(version=SELECTION_VERSION, hash="SHA-256", encoding="UTF-8",
                       payload="stage3-dev-v1:{place|image}:{seed}\\0{key}",
                       ordering="digest ascending, key ascending; places first, then one SOURCE per place",
                       selection_inputs="identities only; no images, masks or retrieval outcomes"),
        inputs_sha256=input_hashes, split_path=str(split_path), split_sha256=split_hash,
        images_root=str(images_root), source_place_count=len(counts), source_image_count=len(sources),
        eligible_source_place_count=len(counts) - len(stage2_places),
        stage2_audit=dict(cohorts=cohort_audits, excluded_place_count=len(stage2_places),
                          excluded_place_keys=sorted(stage2_places),
                          seed0_seed1_overlap=sorted(seed_places[0] & seed_places[1])),
        targets_audit=dict(record_count=target_count, place_count=len(target_places),
                           stage2_overlap_place_keys=sorted(target_places & stage2_places),
                           outside_stage2_place_keys=sorted(target_places - stage2_places)),
        reference_definition=dict(mode="support", role="SUPPORT", scope="full frozen split",
                                  num_images=len(support), num_places=len(counts),
                                  split_sha256=split_hash, cohort_filter_applied=False,
                                  positives="all SUPPORT images with the query place_key",
                                  negatives="all SUPPORT images with any other place_key",
                                  stage2_retrieval_identity=retrieval_identity),
        assertions=dict(one_source_per_place=True, dev_stage2_place_disjoint=True,
                        dev_targets_place_disjoint=True, source_support_image_disjoint=True,
                        full_support_unchanged=True),
        dev_place_keys=[r["place_key"] for r in records],
        cohort_sha256=hashlib.sha256(cohort_text.encode("utf-8")).hexdigest(),
    )
    # Refuse a partial publication if metadata changed during the build.
    if any(_sha256(path) != digest for path, digest in input_hashes.items()):
        raise CohortError("Input metadata changed during cohort build")
    artifacts = {"cohort.jsonl": cohort_text.encode("utf-8"),
                 "summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")}
    if destination.exists():
        # A downstream export does not change the frozen cohort. Never inspect,
        # rewrite or remove its artifacts when verifying an identical rerun.
        entries = {p.name for p in destination.iterdir()} if destination.is_dir() else set()
        if (destination.is_dir() and (destination / "vulnerability").is_dir()
                and not (destination / "vulnerability").is_symlink()):
            entries.discard("vulnerability")
        if not destination.is_dir() or entries != set(artifacts):
            raise CohortError("Existing dev directory differs; refusing to overwrite")
        if any((destination / name).read_bytes() != content for name, content in artifacts.items()):
            raise CohortError("Existing dev cohort differs; refusing to overwrite")
        return summary
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dev-cohort-", dir=destination.parent))
    try:
        for name, content in artifacts.items():
            (staging / name).write_bytes(content)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
