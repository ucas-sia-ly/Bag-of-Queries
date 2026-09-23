"""Fair random translations, complete cohorts and paired statistical units."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2 as T

from src.analysis.perturb import NoLegalTranslation
from src.stage2.data.gsv_split import SplitRecord
from src.stage2.gsv_occlusion import (
    ENDPOINTS, evaluate_query, sample_unique_places, shape_matched_controls,
    stable_seed, summarize_experiment,
)
from src.stage2.retrieval import MEAN, STD, RetrievalContext
from src.stage2.vulnerability import VulnerabilityEstimator


def synthetic_rows(effect_by_seed=None, n=20):
    effects = effect_by_seed or {0: .25, 1: .25}
    rows = []
    for seed, effect in effects.items():
        for strategy in ("attention", "fused"):
            for query in range(n):
                base = dict(seed=seed, strategy=strategy, image_key=f"City/{query}.png", place_key=f"City:{query}",
                            paired_eligible=True, exclusion_reason="", clean_margin=1., clean_rank=1,
                            applied_mask_pixels=10, shift_y=1, shift_x=0)
                conditions = [("clean", -1, 0.), ("targeted", -1, .125 + query/64 + effect)]
                conditions += [("random", repeat, .0625 + repeat/32 + query/64) for repeat in range(5)]
                for condition, repeat, damage in conditions:
                    row = dict(base, **{key: 0. for key in ENDPOINTS}, condition=condition, random_repeat=repeat,
                               perturbed_margin=1.-damage)
                    row["margin_drop"] = damage
                    if condition == "clean":
                        row["applied_mask_pixels"] = 0
                    rows.append(row)
    return rows


class Aggregator(nn.Module):
    def forward(self, features, return_head_attn=False):
        attention = features.mean(1).flatten(1).softmax(-1)
        return F.normalize(features.flatten(1), dim=1), [attention[:, None]]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.AdaptiveAvgPool2d((4, 6))
        self.aggregator = Aggregator()
        self.calls = []

    def forward(self, images):
        self.calls.append(len(images))
        return self.aggregator(self.backbone(images))


class GSVOcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_random_masks_are_same_shape_area_and_reproducible_translations(self):
        mask = torch.zeros(4, 6, dtype=torch.bool)
        mask[1:3, 2] = True
        mask[2, 3] = True
        state = torch.get_rng_state().clone()
        target, masks, offsets, count = shape_matched_controls(mask, (12, 24), 5, seed=123)
        target2, masks2, offsets2, count2 = shape_matched_controls(mask, (12, 24), 5, seed=123)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(target, target2) and torch.equal(masks, masks2))
        self.assertEqual(offsets, offsets2)
        self.assertEqual(count, count2)
        self.assertTrue((masks.sum((1, 2)) == 36).all())
        for shifted, (dy, dx) in zip(masks, offsets):
            self.assertNotEqual((dy, dx), (0, 0))
            self.assertTrue(torch.equal(shifted.nonzero(), target.nonzero() + torch.tensor([dy, dx])))
        with self.assertRaises(NoLegalTranslation):
            shape_matched_controls(torch.eye(4, dtype=torch.bool), (8, 8), 5, seed=1)

    def test_sampling_is_one_source_per_place_and_order_independent(self):
        records = [SplitRecord(f"City/{place}_{image}.png", f"City:{place}", "City", place, "SOURCE")
                   for place in range(30) for image in range(3)]
        a = sample_unique_places(SimpleNamespace(source=records), 10, 0)
        b = sample_unique_places(SimpleNamespace(source=records[::-1]), 10, 0)
        self.assertEqual(a, b)
        self.assertEqual(len({r.place_key for r in a}), 10)
        self.assertNotEqual(a, sample_unique_places(SimpleNamespace(source=records), 10, 1))
        self.assertEqual(stable_seed(1, "a", "attention"), stable_seed(1, "a", "attention"))
        self.assertNotEqual(stable_seed(1, "a", "attention"), stable_seed(1, "a", "fused"))

    def test_random_repeats_are_averaged_before_query_bootstrap_and_variance(self):
        result = summarize_experiment(synthetic_rows(), seeds=[0, 1])
        self.assertEqual(result["status"], "GO")
        self.assertEqual(len(result["pairs"]), 80)  # 20 queries × 2 seeds × 2 strategies, not 400 repeats.
        for comparison in result["comparisons"]:
            self.assertEqual(comparison["n_paired"], 20)
            self.assertAlmostEqual(comparison["mean_paired_difference"], .25)
            self.assertAlmostEqual(comparison["ci_low"], .25)
            self.assertAlmostEqual(comparison["ci_high"], .25)
            self.assertLess(comparison["wilcoxon_p_holm"], .05)
        random = next(s for s in result["summaries"] if s["seed"] == 0 and s["strategy"] == "attention"
                      and s["condition"] == "random" and s["endpoint"] == "margin_drop")
        self.assertAlmostEqual(random["mean"], np.mean(.125 + np.arange(20)/64))
        self.assertAlmostEqual(random["variance"], np.var(.125 + np.arange(20)/64, ddof=1))

    def test_two_sided_wilcoxon_and_ci_gate_stop_on_zero_or_inconsistent_effect(self):
        for effects in ({0: 0., 1: 0.}, {0: .25, 1: -.25}):
            result = summarize_experiment(synthetic_rows(effects), seeds=[0, 1])
            self.assertEqual(result["status"], "STOP")
        result = summarize_experiment(synthetic_rows({0: 0., 1: 0.}), seeds=[0, 1])
        self.assertTrue(all(c["wilcoxon_p_raw"] == 1 for c in result["comparisons"]))

    def test_missing_draws_duplicates_area_errors_and_wrong_margin_are_rejected(self):
        rows = synthetic_rows()
        random_index = next(i for i, row in enumerate(rows) if row["condition"] == "random")
        for kind in ("missing", "duplicate", "area", "margin", "offset"):
            corrupt = deepcopy(rows)
            if kind == "missing":
                del corrupt[random_index]
            elif kind == "duplicate":
                corrupt.append(deepcopy(corrupt[random_index]))
            elif kind == "area":
                corrupt[random_index]["applied_mask_pixels"] = 11
            elif kind == "margin":
                corrupt[random_index]["margin_drop"] += .01
            else:
                corrupt[random_index]["shift_y"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_experiment(corrupt, seeds=[0, 1])

    def test_impossible_translations_are_counted_and_low_coverage_blocks_go(self):
        rows = synthetic_rows()
        for row in rows:
            if int(row["place_key"].split(":")[1]) < 5:
                row.update(paired_eligible=False, exclusion_reason="full-image bounding box")
        rows = [r for r in rows if r["paired_eligible"] or r["condition"] != "random"]
        result = summarize_experiment(rows, seeds=[0, 1])
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(len(result["exclusions"]), 20)
        self.assertTrue(all(c["n_total"] == 20 and c["n_paired"] == 15 for c in result["comparisons"]))

    def test_cohort_mismatch_repeated_places_and_inadequate_resampling_are_rejected(self):
        rows = synthetic_rows()
        corrupt = [r for r in rows if not (r["strategy"] == "fused" and r["image_key"] == "City/0.png")]
        with self.assertRaisesRegex(ValueError, "same preselected"):
            summarize_experiment(corrupt, seeds=[0, 1])
        corrupt = deepcopy(rows)
        for row in corrupt:
            if row["place_key"] == "City:1":
                row["place_key"] = "City:0"
        with self.assertRaisesRegex(ValueError, "one SOURCE per place"):
            summarize_experiment(corrupt, seeds=[0, 1])
        with self.assertRaises(ValueError):
            summarize_experiment(rows, seeds=[0, 1], resamples=9999)

    def test_evaluator_batches_all_conditions_and_does_not_drop_zero_damage_queries(self):
        torch.manual_seed(5)
        rgb = torch.rand(3, 8, 12)
        model = Model().eval()
        descriptor, _ = model(T.Normalize(MEAN, STD)(rgb)[None])
        # Equal positive/negative references force every margin and window damage to zero.
        bank = RetrievalContext(descriptor.repeat(2, 1), ["City:1", "Other:1"])
        record = SplitRecord("City/source.png", "City:1", "City", 1, "SOURCE")
        estimator = VulnerabilityEstimator(top_k=4, strict=False)
        rows, diagnostic, masks = evaluate_query(model, rgb, record, bank, estimator, seed=0)
        self.assertEqual(len(rows), 14)
        self.assertEqual(model.calls[-1], 13)
        self.assertEqual(diagnostic["status"], "STOP")
        self.assertTrue(all(r["paired_eligible"] for r in rows))
        self.assertTrue(all(r["margin_drop"] == 0 for r in rows))
        self.assertEqual(set(masks), {"attention", "fused"})
        estimator = VulnerabilityEstimator(top_k=4, mask_ratio=1., strict=False)
        rows, _, _ = evaluate_query(model, rgb, record, bank, estimator, seed=0)
        self.assertEqual(len(rows), 4)  # retain clean/targeted, exclude only impossible random controls
        self.assertTrue(all(not r["paired_eligible"] for r in rows))


if __name__ == "__main__":
    unittest.main()
