"""Run with: python -m unittest discover -s tests -v"""

import unittest

import torch

from src.analysis import (
    aggregate_attention, build_attention_mask, extract_attention_map,
    robust_normalize, upsample_token_mask,
)
from src.boq import BoQ


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(16),
        )
        self.aggregator = BoQ(16, 128, 5, 2, 4)

    def forward(self, images):
        return self.aggregator(self.backbone(images))


class AttentionAnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_aggregation_axes_and_grid(self):
        a = torch.arange(2 * 3 * 5 * 21).reshape(2, 3, 5, 21).float()
        b = a.flip(-1) * 2
        expected = ((a.mean((1, 2)) + b.mean((1, 2))) / 2).reshape(2, 3, 7)
        for layers in ([a, b], [a.mean(1), b.mean(1)], [a, b.mean(1)]):
            torch.testing.assert_close(aggregate_attention(layers, 3, 7), expected)
        with self.assertRaises(AssertionError):
            aggregate_attention([a], 3, 8)

    def test_normalization_is_per_image(self):
        maps = torch.arange(100).reshape(1, 10, 10).float()
        batch = torch.cat([maps, 10 * maps + 500, torch.full_like(maps, 7)])
        result = robust_normalize(batch)
        expected = ((maps[0] - 4.95) / 89.1).clamp(0, 1)
        torch.testing.assert_close(result[0], expected)
        torch.testing.assert_close(result[0], result[1])
        self.assertTrue(torch.equal(result[2], torch.zeros_like(result[2])))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(robust_normalize(maps)[0], result[0])

    def test_extraction_preserves_descriptor_state_and_training_flags(self):
        model = SmallModel().eval()
        images = torch.randn(2, 3, 10, 14)
        with torch.no_grad():
            expected, _ = model(images)
        model.train()
        model.aggregator.boqs[0].eval()  # Deliberately mixed training flags.
        flags = [m.training for m in model.modules()]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        parameters = [p.requires_grad for p in model.parameters()]
        for per_head in [True, False]:
            result = extract_attention_map(model, images, return_head_attn=per_head, return_attentions=True)
            torch.testing.assert_close(result["descriptor"], expected, rtol=0, atol=0)
            self.assertEqual(result["token_grid"], (5, 7))
            self.assertEqual(result["token_attention_map"].shape, (2, 5, 7))
            self.assertEqual(result["pixel_attention_map"].shape, (2, 10, 14))
            self.assertEqual(result["attentions"][0].ndim, 4 if per_head else 3)
            self.assertFalse(result["descriptor"].requires_grad)
            self.assertEqual(flags, [m.training for m in model.modules()])
            self.assertEqual(parameters, [p.requires_grad for p in model.parameters()])
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        def wrong_grid(module, args, output):
            return output[0], [a[..., :-1] for a in output[1]]
        handle = model.aggregator.register_forward_hook(wrong_grid)
        try:
            with self.assertRaises(AssertionError):
                extract_attention_map(model, images)
            self.assertEqual(flags, [m.training for m in model.modules()])
        finally:
            handle.remove()

    def test_exact_budgets_and_raw_ranking(self):
        for height, width in [(23, 23), (3, 7), (1, 1), (1, 9)]:
            scores = torch.rand(2, height, width)
            for mode in ["raw_topk", "connected_topk"]:
                for ratio in [0, .10, .15, .20, 1]:
                    result = build_attention_mask(scores, ratio, mode)
                    budget = round(ratio * height * width)
                    self.assertEqual(result.dtype, torch.bool)
                    self.assertTrue((result.sum((1, 2)) == budget).all())
                    self.assertTrue(torch.equal(result, build_attention_mask(scores, ratio, mode)))
                    if mode == "raw_topk" and 0 < budget < height * width:
                        for values, mask in zip(scores, result):
                            self.assertGreaterEqual(values[mask].min(), values[~mask].max())
        tied = build_attention_mask(torch.ones(3, 7), .20, "raw_topk")
        self.assertEqual(tied.flatten().nonzero().flatten().tolist(), [0, 1, 2, 3])

    def test_connected_frontier_matches_independent_reference(self):
        # Reference scans every unselected cell, independently of the heap code.
        for scores in [torch.rand(5, 7), torch.ones(5, 7)]:
            height, width = scores.shape
            selected = {int(scores.flatten().argmax())}
            for budget in range(1, height * width + 1):
                result = build_attention_mask(scores, budget / (height * width))
                actual = set(result.flatten().nonzero().flatten().tolist())
                self.assertEqual(actual, selected)
                if budget == height * width:
                    break
                frontier = []
                for index in range(height * width):
                    if index in selected:
                        continue
                    r, c = divmod(index, width)
                    if any(max(abs(r - s // width), abs(c - s % width)) == 1 for s in selected):
                        frontier.append(index)
                best = min(frontier, key=lambda i: (-scores.flatten()[i].item(), i))
                selected.add(best)
        diagonal = torch.tensor([[9., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
        result = build_attention_mask(diagonal, 3 / 9)
        self.assertTrue(torch.equal(result, torch.eye(3, dtype=torch.bool)))

    def test_nearest_mask_expansion(self):
        mask = torch.tensor([[True, False, True], [False, True, False]])
        expected = mask.repeat_interleave(3, 0).repeat_interleave(4, 1)
        self.assertTrue(torch.equal(upsample_token_mask(mask, (6, 12)), expected))
        self.assertEqual(upsample_token_mask(mask[None], (7, 11)).shape, (1, 7, 11))
        self.assertEqual(upsample_token_mask(mask, (7, 11)).dtype, torch.bool)

    def test_invalid_inputs(self):
        scores = torch.ones(3, 7)
        for ratio in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                build_attention_mask(scores, ratio)
        with self.assertRaises(ValueError):
            build_attention_mask(scores, .15, "dilation")
        with self.assertRaises(ValueError):
            build_attention_mask(scores * float("nan"), .15)
        with self.assertRaises(ValueError):
            robust_normalize(scores[None], 95, 5)
        with self.assertRaises(ValueError):
            upsample_token_mask(scores, (10, 14))


if __name__ == "__main__":
    unittest.main()
