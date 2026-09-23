"""Model-free split invariants, reproducibility and command-line integration."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.stage2.data.gsv_split import REQUIRED_COLUMNS, build_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/stage2_build_split.py"


class Stage2SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataframes = self.root / "Dataframes"
        self.dataframes.mkdir()
        self.rows = {}
        for city in ("Alpha", "Beta"):
            rows = []
            for place_id, views in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 12)):
                for view in range(views):
                    rows.append(dict(place_id=place_id, year=2020, month=1, northdeg=view,
                                     city_id=city, lat="13.123456789", lon="100.123456789",
                                     panoid=f"pano_{place_id}_{view}"))
            self.rows[city] = rows
            self.write_city(city, rows)

    def write_city(self, city, rows):
        with (self.dataframes / f"{city}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def build(self, cities=None, seed=0):
        return build_split(cities or ["Alpha", "Beta"], seed, dataframes_dir=self.dataframes)

    def run_cli(self, output_dir, seed=0, hash_seed="0", cities=("Beta", "Alpha")):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--dataframes-dir", str(self.dataframes),
             "--output-dir", str(output_dir), "--seed", str(seed), "--cities", *cities],
            cwd=self.root, env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True, text=True,
        )

    def test_partition_and_exclusion_accounting(self):
        manifest = self.build()
        groups = defaultdict(list)
        for record in manifest.records:
            groups[record.place_key].append(record)
            self.assertEqual(record.place_key, f"{record.city_id}:{record.local_place_id}")
            self.assertTrue(record.image_key.startswith(f"{record.city_id}/{record.city_id}_"))
        keys = [record.image_key for record in manifest.records]
        self.assertEqual(len(keys), len(set(keys)))
        source = {r.image_key for r in manifest.records if r.role == "SOURCE"}
        support = {r.image_key for r in manifest.records if r.role == "SUPPORT"}
        self.assertFalse(source & support)
        self.assertEqual(source | support, set(keys))
        self.assertEqual(len(groups), 6)
        for place_key, records in groups.items():
            roles = Counter(r.role for r in records)
            self.assertEqual(roles["SUPPORT"], 2)
            self.assertGreaterEqual(roles["SOURCE"], 1)
            self.assertNotIn(int(place_key.split(":")[1]), (1, 2))
        self.assertEqual(manifest.summary["totals"], dict(
            input_rows=44, unique_images=44, duplicate_rows=0, input_places=10,
            kept_places=6, excluded_places=4, excluded_images=6, output_images=38,
            support_images=12, source_images=26,
        ))
        self.assertEqual(manifest.summary["quality_gate"]["status"], "GO")

    def test_same_seed_is_identical_and_different_seed_changes_roles(self):
        first = self.build(seed=0)
        self.assertEqual(first, self.build(seed=0))
        second = self.build(seed=42)
        self.assertNotEqual(first.summary["manifest_sha256"], second.summary["manifest_sha256"])
        self.assertNotEqual({r.image_key: r.role for r in first.records},
                            {r.image_key: r.role for r in second.records})
        self.assertEqual(first.summary["totals"], second.summary["totals"])

    def test_input_order_and_unrelated_cities_do_not_change_assignments(self):
        original = self.build()
        self.assertEqual(original, self.build(["Beta", "Alpha", "Alpha"]))
        self.assertEqual(tuple(r for r in original.records if r.city_id == "Alpha"),
                         self.build(["Alpha"]).records)
        for city, rows in self.rows.items():
            self.write_city(city, list(reversed(rows)))
        reordered = self.build()
        self.assertEqual(original.records, reordered.records)
        self.assertEqual(original.summary["manifest_sha256"], reordered.summary["manifest_sha256"])
        # Raw CSV hashes intentionally record file changes, including reordering.
        self.assertNotEqual(original.summary["input_csv_sha256"], reordered.summary["input_csv_sha256"])

    def test_duplicate_rows_cannot_supply_support_or_make_a_place_valid(self):
        original = self.build()
        rows = self.rows["Alpha"]
        self.write_city("Alpha", rows + [rows[0]] * 4 + [rows[-1]] * 2)
        result = self.build()
        self.assertEqual(original.records, result.records)
        self.assertEqual(result.summary["totals"]["duplicate_rows"], 6)
        self.assertEqual(result.summary["per_city"]["Alpha"]["excluded_local_place_ids"], [1, 2])

    def test_heading_distinguishes_images_from_the_same_panorama(self):
        rows = [dict(self.rows["Alpha"][0], northdeg=heading) for heading in (-2, 90, 536)]
        self.write_city("Alpha", rows)
        result = self.build(["Alpha"])
        self.assertEqual(len(result.records), 3)
        self.assertEqual(len({r.image_key for r in result.records}), 3)
        self.assertEqual(result.summary["totals"]["source_images"], 1)
        self.assertTrue(any("_-02_" in r.image_key for r in result.records))
        self.assertTrue(any("_536_" in r.image_key for r in result.records))

    def test_sort_matches_sha256_protocol(self):
        groups = defaultdict(list)
        for record in self.build(seed=9).records:
            groups[record.place_key].append(record)
        for records in groups.values():
            keys = [r.image_key for r in records]
            expected = sorted(keys, key=lambda key: (hashlib.sha256(("9\0" + key).encode()).digest(), key))
            self.assertEqual(keys, expected)
            self.assertEqual([r.role for r in records[:2]], ["SUPPORT", "SUPPORT"])
            self.assertTrue(all(r.role == "SOURCE" for r in records[2:]))

    def test_cli_is_byte_identical_across_python_hash_seeds_and_working_dirs(self):
        outputs = [self.root / "first", self.root / "second"]
        for output, hash_seed in zip(outputs, ("1", "98765")):
            result = self.run_cli(output, hash_seed=hash_seed)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GO:", result.stdout)
        for filename in ("gsv_split.jsonl", "gsv_split_summary.json"):
            self.assertEqual((outputs[0] / filename).read_bytes(), (outputs[1] / filename).read_bytes())
        data = (outputs[0] / "gsv_split.jsonl").read_bytes()
        rows = [json.loads(line) for line in data.splitlines()]
        summary = json.loads((outputs[0] / "gsv_split_summary.json").read_text())
        self.assertEqual(len(rows), summary["totals"]["output_images"])
        self.assertEqual(set(rows[0]), {"image_key", "place_key", "city_id", "local_place_id", "role"})
        self.assertEqual(hashlib.sha256(data).hexdigest(), summary["manifest_sha256"])

    def test_stop_when_a_city_has_majority_invalid_places(self):
        self.write_city("Alpha", [row for row in self.rows["Alpha"] if row["place_id"] <= 3])
        with self.assertRaisesRegex(ValueError, "STOP: Alpha: 2/3"):
            self.build()
        output = self.root / "stopped"
        result = self.run_cli(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STOP: Alpha", result.stderr)
        self.assertFalse(output.exists())

    def test_exactly_half_invalid_is_allowed(self):
        self.write_city("Alpha", [row for row in self.rows["Alpha"] if row["place_id"] <= 4])
        self.assertEqual(self.build(["Alpha"]).summary["per_city"]["Alpha"]["excluded_place_fraction"], .5)

    def test_empty_city_and_only_invalid_places_stop(self):
        for rows in ([], self.rows["Alpha"][:3]):
            self.write_city("Alpha", rows)
            with self.assertRaisesRegex(ValueError, "STOP"):
                self.build(["Alpha"])

    def test_missing_columns_metadata_and_mismatched_city_fail(self):
        for field, value in (("panoid", ""), ("city_id", "Beta"), ("place_id", "oops"),
                             ("month", "0"), ("lat", "")):
            with self.subTest(field=field):
                self.write_city("Alpha", [dict(self.rows["Alpha"][0], **{field: value})])
                with self.assertRaisesRegex(ValueError, "Alpha.csv:2"):
                    self.build(["Alpha"])
        (self.dataframes / "Alpha.csv").write_text("place_id,year\n1,2020\n")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.build(["Alpha"])
        with self.assertRaises(FileNotFoundError):
            self.build(["Missing"])

    def test_invalid_arguments(self):
        for cities in ([], "Alpha", ["../Alpha"], ["Alpha:1"]):
            with self.assertRaises(ValueError):
                build_split(cities, dataframes_dir=self.dataframes)
        for seed in (1.5, "0", True):
            with self.assertRaises(TypeError):
                self.build(seed=seed)

    def test_no_model_library_imported(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import src.stage2.data.gsv_split; "
             "assert 'torch' not in sys.modules; assert 'src.model' not in sys.modules"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
