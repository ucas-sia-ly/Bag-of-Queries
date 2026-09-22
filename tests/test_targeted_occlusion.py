import unittest

import numpy as np
import torch

from src.analysis.perturb import NoLegalTranslation, make_shape_matched_random_mask, perturb_rgb
from src.analysis.retrieval import retrieval_metrics, paired_query_rows, summarize, ENDPOINTS


class TargetedOcclusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_exact_translation_and_repeatability(self):
        mask = torch.zeros(9, 11, dtype=torch.bool)
        mask[2:5, 3] = True
        mask[4, 3:7] = True
        draws, offsets, count = make_shape_matched_random_mask(mask, seed=2024)
        again, same_offsets, _ = make_shape_matched_random_mask(mask, seed=2024)
        self.assertTrue(torch.equal(draws, again))
        self.assertEqual(offsets, same_offsets)
        self.assertEqual(count, (9 - 3 + 1) * (11 - 4 + 1) - 1)
        for draw, (dy, dx) in zip(draws, offsets):
            shifted = mask.nonzero() + torch.tensor([dy, dx])
            self.assertTrue(torch.equal(draw.nonzero(), shifted))
            self.assertEqual(int(draw.sum()), int(mask.sum()))
            self.assertFalse(torch.equal(draw, mask))

    def test_only_one_legal_alternative_and_impossible_mask(self):
        mask = torch.ones(4, 3, dtype=torch.bool)
        mask[-1] = False
        draws, offsets, count = make_shape_matched_random_mask(mask, seed=0)
        self.assertEqual(count, 1)
        self.assertEqual(offsets, [(1, 0)] * 5)
        self.assertTrue(torch.equal(draws[0], draws[-1]))
        for impossible in [torch.ones(3, 3, dtype=torch.bool), torch.eye(3, dtype=torch.bool), torch.zeros(3, 3, dtype=torch.bool)]:
            with self.assertRaises(NoLegalTranslation):
                make_shape_matched_random_mask(impossible, seed=0)

    def test_mean_fill_rgb_and_blur(self):
        rgb = torch.rand(3, 9, 11)
        original = rgb.clone()
        masks = torch.zeros(2, 9, 11, dtype=torch.bool)
        masks[0, 1:3, 2:4] = True
        masks[1, 5:7, 7:9] = True
        result = perturb_rgb(rgb, masks)
        for value, mask in zip(result, masks):
            torch.testing.assert_close(value[:, mask], rgb.mean((1, 2))[:, None].expand(-1, 4))
            self.assertTrue(torch.equal(value[:, ~mask], rgb[:, ~mask]))
        self.assertTrue(torch.equal(original, rgb))
        blurred = perturb_rgb(rgb, masks, operator="gaussian_blur", blur_kernel=3, blur_sigma=1.)
        self.assertTrue(torch.equal(blurred[0, :, ~masks[0]], rgb[:, ~masks[0]]))
        with self.assertRaises(ValueError):
            perturb_rgb(rgb - 2, masks)

    def test_full_database_metrics_against_numpy_sort(self):
        references = torch.nn.functional.normalize(torch.randn(30, 8), dim=1)
        queries = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        positives = np.array([2, 4, 19])
        metrics = retrieval_metrics(queries, references, positives, queries[0])
        similarities = (queries @ references.T).numpy()
        for row, scores in zip(metrics, similarities):
            ordered = np.lexsort((np.arange(len(references)), -scores))
            rank = next(i + 1 for i, index in enumerate(ordered) if index in positives)
            self.assertEqual(row["rank"], rank)
            self.assertAlmostEqual(row["positive_sim"], float(scores[positives].max()))
            self.assertAlmostEqual(row["negative_sim"], float(np.delete(scores, positives).max()))
            self.assertEqual(row["hit_at_5"], int(rank <= 5))
        # Exact ties use reference index, including when positive scores tie.
        tied = retrieval_metrics(torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]] * 4), [2, 3], torch.tensor([1., 0.]))
        self.assertEqual(tied[0]["rank"], 3)
        self.assertEqual(tied[0]["margin"], 0)

    def test_query_level_random_averaging_and_exclusions(self):
        rows = []
        for q in range(3):
            base = dict(query_id=q, mask_ratio=.15, clean_rank=1, paired_eligible=q != 2)
            target = dict(base, condition="attention", random_repeat=-1, **{key: 2. for key in ENDPOINTS})
            rows.append(target)
            if q < 2:
                for repeat in range(5):
                    rows.append(dict(base, condition="random", random_repeat=repeat,
                                     **{key: float(repeat) for key in ENDPOINTS}))
        pairs = paired_query_rows(rows, .15)
        self.assertEqual(len(pairs), 2)  # not ten independent observations
        self.assertEqual(pairs[0]["random_margin_drop"], 2.)
        summary, _ = summarize(rows, [.15], bootstrap_samples=20)
        self.assertEqual(summary[0]["n_excluded_no_translation"], 1)
        self.assertEqual(summary[0]["paired_margin_difference"], 0.)
        with self.assertRaises(ValueError):
            paired_query_rows(rows[:-2] + [rows[-1]], .15)


if __name__ == "__main__":
    unittest.main()
