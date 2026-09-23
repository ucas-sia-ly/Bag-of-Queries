"""Small fixed-feature, cache-integrity and original-trainer integration tests."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import write_png

from scripts.stage2_build_retrieval import trainer_recalls, validate_context
from src.stage2.data.gsv_split import SplitRecord
from src.stage2.retrieval import (
    GSVImages, GSVSplit, MEAN, STD, RetrievalContext, cache_identity, encode_images,
    evaluation_transform, get_support_cache, load_support_cache, place_prototypes,
    validate_descriptors,
)


class DummyBoQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer("features", torch.tensor([
            [1., 0.], [.8, .6], [0., 1.], [-.6, .8], [-1., 0.], [-.8, -.6],
        ]))

    def forward(self, images):
        index = ((images[:, 0, 0, 0] * STD[0] + MEAN[0]) * 255 / 40).round().long()
        return self.features[index] * self.scale, []


class Stage2RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "检索缓存"
        self.images = self.root / "Images"
        (self.images / "City").mkdir(parents=True)
        self.records = []
        for index in range(6):
            place = index // 3
            record = SplitRecord(f"City/{index}.png", f"City:{place}", "City", place,
                                 "SOURCE" if index % 3 == 2 else "SUPPORT")
            self.records.append(record)
            image = torch.zeros(3, 12, 16, dtype=torch.uint8)
            image[0] = 40 * index
            write_png(image, str(self.images / record.image_key))
        self.manifest = self.root / "split.jsonl"
        self.write_manifest(self.records)
        self.split = GSVSplit.read(self.manifest)
        self.model = DummyBoQ()
        self.checkpoint = self.root / "weights.pth"
        self.checkpoint.write_bytes(b"dummy frozen weights")
        self.identity = cache_identity(self.split, self.images, self.checkpoint,
                                       {"descriptor_dim": 2}, [8, 8], 3, "cpu")
        self.cache = self.root / "support.pt"

    def write_manifest(self, records):
        self.manifest.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))

    def build_cache(self):
        return get_support_cache(self.cache, self.model, self.split, self.images, self.identity)

    def test_dummy_encoding_full_database_ranking_and_trainer_agree(self):
        descriptors, reused = self.build_cache()
        self.assertFalse(reused)
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.scale.requires_grad)
        torch.testing.assert_close(descriptors, self.model.features[[0, 1, 3, 4]])
        queries = encode_images(self.model, GSVImages(self.split.source, self.images, [8, 8]))
        context = RetrievalContext.from_support(descriptors, self.split.support)
        self.assertEqual(context.positive_indices("City:0"), (0, 1))
        self.assertEqual(context.negative_indices("City:0").tolist(), [2, 3])
        rows, metrics = validate_context(context, queries, self.split.source, self.root)
        self.assertEqual([r["rank"] for r in rows], [2, 1])
        self.assertAlmostEqual(rows[0]["positive_sim"], .6)
        self.assertAlmostEqual(rows[0]["negative_sim"], .8)
        self.assertAlmostEqual(rows[0]["margin"], -.2)
        self.assertAlmostEqual(rows[1]["margin"], 1.6, places=6)
        self.assertEqual(metrics["recalls"], {"R@1": .5, "R@5": 1., "R@10": 1.})
        self.assertEqual(metrics["recalls"], metrics["trainer_recalls"])
        self.assertLess(metrics["oracle_max_abs_error"], 1e-6)

    def test_preprocessing_is_identical_to_original_datamodule_validation(self):
        from src.dataloaders.datamodule import VPRDataModule
        pixels = torch.randint(0, 256, (3, 19, 21), dtype=torch.uint8)
        original = VPRDataModule(val_img_size=(14, 14)).val_transform(pixels)
        torch.testing.assert_close(evaluation_transform((14, 14))(pixels), original, rtol=0, atol=0)

    def test_prototype_is_normalized_support_mean_and_has_all_global_negatives(self):
        descriptors, _ = self.build_cache()
        keys, prototypes = place_prototypes(descriptors, self.split.support)
        self.assertEqual(keys, ["City:0", "City:1"])
        expected = F.normalize(descriptors.reshape(2, 2, 2).mean(1), dim=1)
        torch.testing.assert_close(prototypes, expected)
        torch.testing.assert_close(prototypes.norm(dim=1), torch.ones(2))
        context = RetrievalContext.from_support(descriptors, self.split.support, mode="prototype")
        self.assertEqual(context.positive_indices("City:1"), (1,))
        self.assertEqual(context.negative_indices("City:1").tolist(), [0])
        # Noncontiguous support ordering must preserve place grouping.
        permutation = [0, 2, 1, 3]
        reordered = RetrievalContext.from_support(descriptors[permutation],
                                                  [self.split.support[i] for i in permutation])
        self.assertEqual(reordered.positive_indices("City:0"), (0, 2))
        same_keys, same = place_prototypes(descriptors[permutation], [self.split.support[i] for i in permutation])
        self.assertEqual(keys, same_keys)
        torch.testing.assert_close(same, prototypes)

    def test_cache_reload_never_encodes_again_and_rejects_identity_changes(self):
        first, _ = self.build_cache()
        with patch.object(self.model, "forward", side_effect=AssertionError("must reuse cache")):
            second, reused = self.build_cache()
        self.assertTrue(reused)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for field, value in (("split_sha256", "changed"), ("checkpoint_sha256", "changed"),
                             ("image_size", [16, 16]), ("batch_size", 8)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                load_support_cache(self.cache, self.split, dict(self.identity, **{field: value}))
        self.checkpoint.write_bytes(b"updated weights")
        changed = cache_identity(self.split, self.images, self.checkpoint, {"descriptor_dim": 2}, [8, 8], 3, "cpu")
        self.assertNotEqual(self.identity, changed)

    def test_stale_identity_cannot_hide_a_changed_source_split(self):
        self.build_cache()
        records = list(self.records)
        records[2] = replace(records[2], image_key="City/changed_source.png")
        self.write_manifest(records)
        changed_split = GSVSplit.read(self.manifest)
        with self.assertRaisesRegex(ValueError, "current split"):
            load_support_cache(self.cache, changed_split, self.identity)

    def test_cache_rejects_wrong_order_nonfinite_zero_and_wrong_shape(self):
        self.build_cache()
        for corruption in ("order", "place", "nan", "zero", "shape", "dtype"):
            with self.subTest(corruption=corruption):
                payload = torch.load(self.cache, weights_only=True)
                if corruption == "order":
                    payload["image_keys"] = list(reversed(payload["image_keys"]))
                elif corruption == "place":
                    payload["place_keys"][0] = "wrong"
                elif corruption == "nan":
                    payload["descriptors"][0, 0] = float("nan")
                elif corruption == "zero":
                    payload["descriptors"][0] = 0
                elif corruption == "shape":
                    payload["descriptors"] = payload["descriptors"][:-1]
                else:
                    payload["descriptors"] = payload["descriptors"].half()
                corrupt = self.root / "corrupt.pt"
                torch.save(payload, corrupt)
                with self.assertRaises(ValueError):
                    load_support_cache(corrupt, self.split, self.identity)

    def test_missing_or_corrupt_images_fail_without_publishing_a_cache(self):
        target = self.images / self.split.support[0].image_key
        target.unlink()
        with self.assertRaisesRegex(RuntimeError, "Cannot decode GSV image"):
            self.build_cache()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.cache.with_suffix(".building.raw").exists())
        target.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(RuntimeError, "Cannot decode GSV image"):
            self.build_cache()
        self.assertFalse(self.cache.exists())

    def test_unreadable_saved_cache_is_never_published(self):
        def corrupt_save(payload, stream):
            stream.write(b"truncated archive")

        with patch("src.stage2.retrieval.torch.save", side_effect=corrupt_save):
            with self.assertRaises(RuntimeError):
                self.build_cache()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.cache.with_suffix(".building.pt").exists())
        self.assertFalse(self.cache.with_suffix(".building.raw").exists())

    def test_split_rejects_leakage_or_missing_support_and_accepts_lowercase_roles(self):
        for records in (self.records + [self.records[0]],
                        self.records + [replace(self.records[0], role="SOURCE")],
                        self.records[1:],
                        self.records[:3],
                        [replace(self.records[0], image_key="../escaped.png"), *self.records[1:]],
                        [replace(self.records[0], place_key="Wrong:0"), *self.records[1:]]):
            self.write_manifest(records)
            with self.assertRaises(ValueError):
                GSVSplit.read(self.manifest)
        self.write_manifest([replace(r, role=r.role.lower()) for r in self.records])
        self.assertEqual(GSVSplit.read(self.manifest).support, self.split.support)

    def test_prototype_rejects_sources_and_cancelling_vectors(self):
        descriptors, _ = self.build_cache()
        with self.assertRaisesRegex(ValueError, "SUPPORT"):
            place_prototypes(descriptors, [self.split.source[0], *self.split.support[1:]])
        with self.assertRaisesRegex(ValueError, "SUPPORT"):
            RetrievalContext.from_support(descriptors, [self.split.source[0], *self.split.support[1:]])
        descriptors[1] = -descriptors[0]
        with self.assertRaisesRegex(ValueError, "cancels to zero"):
            place_prototypes(descriptors, self.split.support)

    def test_variants_require_clean_and_use_existing_tie_order(self):
        references = torch.tensor([[1., 0.]] * 4)
        context = RetrievalContext(references, ["other", "other", "query", "query"])
        variants = references[:2]
        with self.assertRaisesRegex(ValueError, "explicit clean_descriptor"):
            context.query(variants, "query")
        metrics = context.query(variants, "query", clean_descriptor=references[0])
        self.assertEqual([r["rank"] for r in metrics], [3, 3])
        self.assertEqual([r["margin"] for r in metrics], [0., 0.])
        with self.assertRaisesRegex(ValueError, "No SUPPORT"):
            context.query(variants[0], "missing")
        with self.assertRaisesRegex(ValueError, "unit L2"):
            context.query(variants[0] * 2, "query")
        with self.assertRaisesRegex(ValueError, "two places"):
            RetrievalContext(references, ["only"] * 4)

    def test_original_trainer_mismatch_stops_validation(self):
        descriptors, _ = self.build_cache()
        context = RetrievalContext.from_support(descriptors, self.split.support)
        queries = self.model.features[[2, 5]]
        with patch("scripts.stage2_build_retrieval.trainer_recalls", return_value={"R@1": 0.}):
            with self.assertRaisesRegex(ValueError, "STOP"):
                validate_context(context, queries, self.split.source, self.root)

    def test_source_sampling_is_stable_and_only_sources(self):
        selected = self.split.sample_sources(2, seed=123)
        self.assertEqual(selected, self.split.sample_sources(2, seed=123))
        self.assertTrue(all(r.role == "SOURCE" for r in selected))
        with self.assertRaises(ValueError):
            self.split.sample_sources(3)


if __name__ == "__main__":
    unittest.main()
