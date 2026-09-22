import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_targeted_occlusion import build_ablation_variants
from src.analysis.masks import upsample_token_mask
from src.analysis.perturb import make_token_aligned_random_mask, NoLegalTranslation, make_shape_matched_random_mask
from src.analysis.retrieval import ENDPOINTS, summarize_ablations, ablation_contrasts


class OcclusionAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_token_translation_topology_alignment_and_unique_draws(self):
        # Disconnected shape: translation must preserve ALL components, not grow a new mask.
        mask = torch.zeros(9, 11, dtype=torch.bool)
        mask[2:4, 3] = True
        mask[5, 6] = True
        draws, offsets, count = make_token_aligned_random_mask(mask, seed=42)
        repeat, again, _ = make_token_aligned_random_mask(mask, seed=42)
        self.assertTrue(torch.equal(draws, repeat))
        self.assertEqual(offsets, again)
        self.assertGreaterEqual(count, 5)
        self.assertEqual(len(set(offsets)), 5)
        self.assertNotIn((0, 0), offsets)
        target_pixels = upsample_token_mask(mask, (126, 154))
        pixels = upsample_token_mask(draws, (126, 154))
        for draw, image, (dy, dx) in zip(draws, pixels, offsets):
            self.assertTrue(torch.equal(draw.nonzero(), mask.nonzero() + torch.tensor([dy, dx])))
            self.assertTrue(torch.equal(image.nonzero(), target_pixels.nonzero() + torch.tensor([dy * 14, dx * 14])))
            self.assertEqual(int(draw.sum()), int(mask.sum()))
            self.assertEqual(int(image.sum()), int(target_pixels.sum()))
            self.assertTrue(torch.equal(image, draw.repeat_interleave(14, 0).repeat_interleave(14, 1)))

    def test_fewer_than_five_positions_and_no_position(self):
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[3:] = False
        draws, offsets, count = make_token_aligned_random_mask(mask, seed=1)
        self.assertEqual(count, 2)
        self.assertEqual(set(offsets), {(1, 0), (2, 0)})
        self.assertEqual(len(set(offsets[:2])), 2)
        self.assertEqual(len(draws), 5)
        for mask in [torch.eye(5, dtype=torch.bool), torch.zeros(5, 5, dtype=torch.bool)]:
            with self.assertRaises(NoLegalTranslation):
                make_token_aligned_random_mask(mask, seed=2024)

    def test_variants_equal_budgets_and_preserve_legacy_pixel_sampling(self):
        args = SimpleNamespace(mask_modes=["connected_topk", "raw_topk"], ratios=[.15], seed=2024,
                               random_repeats=5, random_baselines=["random_pixel", "random_token"],
                               primary_baseline="random_token", fill_source="image_mean",
                               operator="mean_fill", blur_kernel=3, blur_sigma=1.)
        rgb = torch.rand(3, 70, 98)
        scores = torch.arange(35).reshape(1, 5, 7).float()
        images, specs = build_ablation_variants(rgb, scores, 17, args)
        targets = [s for s in specs if s["condition"].startswith("attention_")]
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["mask_tokens"], targets[1]["mask_tokens"])
        self.assertEqual(targets[0]["mask_pixels"], targets[1]["mask_pixels"])
        for mode in args.mask_modes:
            target = next(s for s in targets if s["mask_mode"] == mode)
            for spec in [s for s in specs if s["mask_mode"] == mode]:
                self.assertEqual(spec["mask_pixels"], target["mask_pixels"])
                if spec["condition"] == "random_token":
                    self.assertEqual(spec["mask_tokens"], target["mask_tokens"])
                    self.assertEqual(spec["offset"], tuple(v * 14 for v in spec["token_offset"]))
        # Recover the actual erased support and compare with the original pixel sampler.
        target_index = next(i for i, s in enumerate(specs) if s["condition"] == "attention_connected")
        support = (images[target_index] != rgb).any(0)
        seed = int(np.random.SeedSequence([2024, 17, 150000]).generate_state(1)[0])
        expected, offsets, _ = make_shape_matched_random_mask(support, seed=seed)
        actual = [(image != rgb).any(0) for image, s in zip(images, specs)
                  if s["condition"] == "random_pixel" and s["mask_mode"] == "connected_topk"]
        self.assertTrue(torch.equal(torch.stack(actual), expected))
        with self.assertRaises(ValueError):
            build_ablation_variants(torch.rand(3, 71, 98), scores, 17, args)

    def test_raw_untranslatable_retained_and_cohorts_not_mixed(self):
        rows = []
        for mode, target in [("connected_topk", "attention_connected"), ("raw_topk", "attention_raw_topk")]:
            for q in [1, 2]:
                eligible = mode == "connected_topk" or q == 1
                common = dict(query_id=q, mask_mode=mode, mask_ratio=.15, clean_rank=1,
                              mask_tokens=5, mask_pixels=980, eligible_random_token=eligible,
                              eligible_random_pixel=eligible)
                rows.append(dict(common, condition=target, random_repeat=-1,
                                 **{endpoint: 10. if mode == "raw_topk" else 4. for endpoint in ENDPOINTS}))
                if eligible:
                    for baseline, value in [("random_token", 2.), ("random_pixel", 1.)]:
                        for repeat in range(5):
                            rows.append(dict(common, condition=baseline, random_repeat=repeat,
                                             **{endpoint: value for endpoint in ENDPOINTS}))
        summary, pairs = summarize_ablations(rows, [.15], ["connected_topk", "raw_topk"],
                                            ["random_pixel", "random_token"], bootstrap_samples=10)
        self.assertEqual(len(pairs), 6)  # connected 2*2 + raw 1*2, not pooled across modes
        raw = next(s for s in summary if s["condition"] == "attention_raw_topk" and s["baseline"] == "random_token")
        self.assertEqual(raw["n_paired_queries"], 1)
        self.assertEqual(raw["n_excluded_no_translation"], 1)
        self.assertEqual(raw["paired_margin_difference"], 8.)
        contrasts, contrasts_by_query = ablation_contrasts(rows, pairs, [.15], bootstrap_samples=10)
        construction = next(c for c in contrasts if c["comparison"].startswith("attention_raw"))
        self.assertEqual(construction["n_paired_queries"], 2)  # Includes untranslatable raw mask!
        self.assertEqual(construction["margin_difference"], 6.)
        alignment = next(c for c in contrasts if c["comparison"].startswith("raw_topk"))
        self.assertEqual(alignment["n_paired_queries"], 1)
        self.assertEqual(alignment["margin_difference"], 1.)


if __name__ == "__main__":
    unittest.main()
