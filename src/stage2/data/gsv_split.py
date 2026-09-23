"""Reproducible, image-disjoint GSV-Cities Source/Support manifests.

Each place shares exactly two SUPPORT views among all its SOURCE views. Places
with fewer than three distinct images are excluded. No model or image decoding
is needed. ``image_key`` is a path relative to the dataset's Images directory,
including northdeg to distinguish views with different headings.
"""

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import List


DEFAULT_DATAFRAMES_DIR = Path(__file__).resolve().parents[3] / "data/train/gsv-cities/Dataframes"
K_SUPPORT = 2
REQUIRED_COLUMNS = ("place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid")


@dataclass(frozen=True)
class SplitRecord:
    image_key: str
    place_key: str
    city_id: str
    local_place_id: int
    role: str


def _jsonl_line(record: SplitRecord) -> bytes:
    return (json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass(frozen=True)
class SplitManifest:
    records: tuple[SplitRecord, ...]
    summary: dict

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write deterministic UTF-8 artifacts (no timestamps or absolute paths)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "gsv_split.jsonl"
        summary_path = output_dir / "gsv_split_summary.json"
        with manifest_path.open("wb") as stream:
            for record in self.records:
                stream.write(_jsonl_line(record))
        summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path, summary_path


def _image_identity(row: dict, city: str) -> tuple[int, str]:
    if any(row.get(field) is None or not row[field].strip() for field in REQUIRED_COLUMNS):
        raise ValueError("Missing required image metadata")
    row = {field: row[field].strip() for field in REQUIRED_COLUMNS}
    if row["city_id"] != city:
        raise ValueError(f"city_id {row['city_id']!r} does not match CSV city {city!r}")
    place_id = int(row["place_id"])
    year, month, northdeg = (int(row[field]) for field in ("year", "month", "northdeg"))
    if place_id < 0 or year < 0 or not 1 <= month <= 12:
        raise ValueError("Invalid place_id/year/month")
    # GSV CSVs contain negative and >360 headings; preserve them in filenames.
    # Match the existing GSV image naming convention, preserving CSV coordinates.
    filename = (
        f"{city}_{place_id:07d}_{year:04d}_{month:02d}_{northdeg:03d}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )
    if "/" in filename or "\\" in filename:
        raise ValueError("Image metadata must not contain path separators")
    return place_id, f"{city}/{filename}"


def build_split(
    cities: List[str], seed: int = 0, *, dataframes_dir: str | Path | None = None,
) -> SplitManifest:
    """Build a split from city CSVs; the default path is relative to this repo.

    Cities and local place IDs are sorted. Within a place, order by
    SHA256(UTF8(str(seed) + NUL + image_key)), breaking digest ties by image_key.
    The first two distinct images become SUPPORT, the rest SOURCE. Reordering
    CSV rows, city arguments or Python's hash seed cannot change the manifest.
    Changing the split seed may change assignments, but tiny places can coincide.

    Duplicate images count once. Missing/malformed metadata, empty cities or a
    city with a majority (>50%) of excluded places cause a STOP (ValueError).
    ``dataframes_dir`` allows another checkout/dataset without changing defaults.
    """
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if isinstance(cities, str) or not cities:
        raise ValueError("cities must be a non-empty list of CSV city names")
    if any(not isinstance(city, str) or not city or city in (".", "..")
           or any(char in city for char in ("/", "\\", ":", "\0")) for city in cities):
        raise ValueError("cities must contain plain CSV city names")
    cities = sorted(set(cities))
    root = Path(dataframes_dir) if dataframes_dir is not None else DEFAULT_DATAFRAMES_DIR
    records = []
    per_city = {}
    input_csv_sha256 = {}
    seed_prefix = f"{seed}\0".encode("utf-8")

    for city in cities:
        path = root / f"{city}.csv"
        # Read one snapshot so provenance and parsed content always agree.
        csv_bytes = path.read_bytes()
        input_csv_sha256[city] = hashlib.sha256(csv_bytes).hexdigest()
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {REQUIRED_COLUMNS}")
        places = defaultdict(set)
        input_rows = 0
        for row in reader:
            input_rows += 1
            try:
                place_id, image_key = _image_identity(row, city)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
            places[place_id].add(image_key)

        excluded_ids = sorted(pid for pid, images in places.items() if len(images) < K_SUPPORT + 1)
        unique_images = sum(map(len, places.values()))
        excluded_images = sum(len(places[pid]) for pid in excluded_ids)
        kept_places = len(places) - len(excluded_ids)
        counts = dict(
            input_rows=input_rows,
            unique_images=unique_images,
            duplicate_rows=input_rows - unique_images,
            input_places=len(places),
            kept_places=kept_places,
            excluded_places=len(excluded_ids),
            excluded_images=excluded_images,
            output_images=unique_images - excluded_images,
            support_images=kept_places * K_SUPPORT,
            source_images=unique_images - excluded_images - kept_places * K_SUPPORT,
        )
        per_city[city] = {
            **counts,
            "excluded_place_fraction": len(excluded_ids) / len(places) if places else 1.0,
            "excluded_local_place_ids": excluded_ids,
            "unique_views_per_place": dict(sorted(Counter(map(len, places.values())).items())),
        }
        if not kept_places or len(excluded_ids) > len(places) / 2:
            raise ValueError(
                f"STOP: {city}: {len(excluded_ids)}/{len(places)} places have fewer than "
                "3 distinct images, or no valid places remain; review the city selection "
                "or support protocol before proceeding (K_support remains 2)."
            )
        for place_id in sorted(places):
            if len(places[place_id]) < K_SUPPORT + 1:
                continue
            ordered = sorted(places[place_id], key=lambda key: (
                hashlib.sha256(seed_prefix + key.encode("utf-8")).digest(), key,
            ))
            records.extend(
                SplitRecord(image_key, f"{city}:{place_id}", city, place_id,
                            "SUPPORT" if index < K_SUPPORT else "SOURCE")
                for index, image_key in enumerate(ordered)
            )

    totals = {field: sum(stats[field] for stats in per_city.values()) for field in counts}
    digest = hashlib.sha256()
    for record in records:
        digest.update(_jsonl_line(record))
    summary = {
        "schema_version": 1,
        "seed": seed,
        "cities": cities,
        "k_support": K_SUPPORT,
        "min_unique_views_per_place": K_SUPPORT + 1,
        "image_key_format": "path relative to Images/: city/city_place_year_month_northdeg_lat_lon_panoid.jpg",
        "ordering": "city ascending, local_place_id ascending, SHA256(str(seed) + NUL + image_key), image_key",
        "exclusion_reason": "fewer than 3 distinct image keys",
        "input_csv_sha256": input_csv_sha256,
        "manifest_sha256": digest.hexdigest(),
        "totals": totals,
        "per_city": per_city,
        "quality_gate": {
            "status": "GO",
            "min_support_per_source_place": K_SUPPORT,
            "source_support_disjoint": True,
            "max_excluded_place_fraction": 0.5,
        },
    }
    return SplitManifest(tuple(records), summary)
