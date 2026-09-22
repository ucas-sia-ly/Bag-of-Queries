import math
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2 as T

from src.analysis.intervention import (
    bootstrap_summary, probe_query, project_window_values, summarize_faithfulness,
    summarize_query, token_windows,
)
from src.analysis.masks import upsample_token_mask
from src.analysis.perturb import perturb_rgb


class ToyAggregator(nn.Module):
    def forward(self, features, return_head_attn=False):
        scores = features.mean(1).flatten(1).softmax(-1)
        return F.normalize(features.flatten(1), dim=1), [scores[:, None]]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.AdaptiveAvgPool2d((3, 5))
        self.aggregator = ToyAggregator()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        return self.aggregator(self.backbone(x))


def example_rows(attention, damage):
    return [dict(query_id=7, window_y=0, window_x=i, attention_score=float(a),
                 margin_drop=float(d), clean_hit_at_1=1) for i, (a, d) in enumerate(zip(attention, damage))]


class InterventionTests(unittest.TestCase):
    def test_rectangular_dynamic_windows_and_equal_budget(self):
        masks, positions = token_windows((4, 6))
        self.assertEqual(masks.shape, (15, 4, 6))
        self.assertEqual(positions.tolist()[0], [0, 0])
        self.assertEqual(positions.tolist()[-1], [2, 4])
        self.assertTrue((masks.sum((1, 2)) == 4).all())
        pixels = upsample_token_mask(masks, (12, 30))
        self.assertTrue((pixels.sum((1, 2)) == 60).all())
        self.assertEqual(len(token_windows((5, 7), 3, 2)[0]), 6)
        with self.assertRaises(ValueError):
            token_windows((1, 5))

    def test_probe_batches_rgb_before_normalize_and_preserves_raw_scores(self):
        torch.manual_seed(2024)
        rgb = torch.rand(3, 6, 10)
        original = rgb.clone()
        normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])
        references = F.normalize(torch.randn(9, 45), dim=1)
        model = ToyModel().eval()
        result = probe_query(model, rgb, normalize, references, [0, 2], 7, batch_size=3)
        self.assertEqual(result["token_grid"], (3, 5))
        self.assertEqual(result["variant_batch_sizes"], [3, 3, 2])
        self.assertEqual(len(model.inputs), math.ceil(8/3))
        self.assertEqual(len(result["rows"]), 8)
        masks, positions = token_windows((3, 5))
        expected = normalize(perturb_rgb(rgb, upsample_token_mask(masks, (6, 10))))
        torch.testing.assert_close(torch.cat(model.inputs), expected)
        self.assertTrue(torch.equal(rgb, original))
        for row, (y, x) in zip(result["rows"], positions.tolist()):
            self.assertAlmostEqual(row["attention_score"], float(result["token_attention_map"][y:y+2, x:x+2].mean()), places=7)
            self.assertEqual(row["margin_drop"], row["clean_margin"]-row["perturbed_margin"])
            self.assertEqual(row["rank_degradation"], row["perturbed_rank"]-row["clean_rank"])
        # Batch partitioning must not alter any retrieval endpoint.
        again = probe_query(ToyModel().eval(), rgb, normalize, references, [0, 2], 7, batch_size=8)
        for a, b in zip(result["rows"], again["rows"]):
            for key in ["margin_drop", "descriptor_drift", "rank_degradation", "positive_similarity_drop"]:
                self.assertAlmostEqual(a[key], b[key], places=6)

    def test_statistics_retain_negative_damage_and_correct_percentile(self):
        q = summarize_query(example_rows([1, 2, 3, 4], [-4, -3, -2, -1]), tail_fraction=.25)
        self.assertEqual(q["spearman"], 1)
        self.assertEqual(q["top_mean_damage"], -1)
        self.assertEqual(q["bottom_mean_damage"], -4)
        self.assertEqual(q["top_minus_bottom_damage"], 3)
        self.assertEqual(q["attention_max_damage_percentile"], 100)
        self.assertEqual(q["negative_damage_window_fraction"], 1)
        q = summarize_query(example_rows([1, 2, 3, 4], [4, 3, 2, 1]), tail_fraction=.25)
        self.assertEqual(q["spearman"], -1)
        self.assertEqual(q["attention_max_damage_percentile"], 0)
        self.assertEqual(q["top_minus_bottom_damage"], -3)

    def test_constant_correlation_is_explicitly_undefined(self):
        q = summarize_query(example_rows([1, 1, 1, 1], [-1, 0, 1, 2]))
        self.assertIsNone(q["spearman"])
        self.assertFalse(q["spearman_defined"])
        summary = summarize_faithfulness([q], resamples=100)
        self.assertIsNone(summary["spearman"])
        self.assertEqual(summary["n_spearman_undefined"], 1)
        self.assertIn("not sufficiently faithful", summary["conclusion"])

    def test_query_bootstrap_and_no_window_pseudoreplication(self):
        q1 = summarize_query(example_rows([1, 2, 3, 4], [1, 2, 3, 4]))
        q2 = dict(q1, query_id=8, spearman=-1, top_minus_bottom_damage=-3)
        result = summarize_faithfulness([q1, q2], seed=6, resamples=500)
        self.assertEqual(result["spearman"]["n_queries"], 2)
        self.assertEqual(result["spearman"]["mean"], 0)
        self.assertEqual(result["spearman"]["positive_fraction"], .5)
        self.assertLessEqual(result["spearman"]["mean_ci_low"], 0)
        self.assertIn("not sufficiently faithful", result["conclusion"])
        with self.assertRaises(ValueError):
            summarize_faithfulness([q1, q1], resamples=100)
        self.assertEqual(bootstrap_summary([1, 2, 3], seed=4, resamples=500),
                         bootstrap_summary([1, 2, 3], seed=4, resamples=500))

    def test_overlap_projection_is_signed_and_has_correct_counts(self):
        projected, counts = project_window_values(np.array([[0, 0], [0, 1]]), [-2., 4.], (2, 3))
        np.testing.assert_array_equal(counts, [[1, 2, 1], [1, 2, 1]])
        np.testing.assert_allclose(projected, [[-2, 1, 4], [-2, 1, 4]])

    def test_stable_but_weak_correlation_is_not_sufficient_faithfulness(self):
        q = summarize_query(example_rows([1, 2, 3, 4], [1, 2, 3, 4]))
        queries = [dict(q, query_id=i, spearman=rho) for i, rho in enumerate([.15, .16, .17])]
        result = summarize_faithfulness(queries, resamples=500)
        self.assertTrue(result["positive_proposal_signal_supported"])
        self.assertIn("not sufficiently faithful", result["conclusion"])


if __name__ == "__main__":
    unittest.main()
