"""Scientific controls: native support, frozen forward, query-level inference."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.analysis.perturb import NoLegalTranslation
from src.stage2.retrieval import RetrievalContext, evaluation_transform
from src.stage3.core_mask_eval import (
    ARMS, _module, choose_core, evaluate_core, exact_controls, freeze_choice,
    native_mean_fill, pair_queries, summarize_pairs,
)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_exact_native_translation_preserves_holes_components_and_reproducibility():
    core = torch.zeros((31, 47), dtype=torch.bool)
    core[8:15, 12:20] = True
    core[10:13, 14:18] = False  # A hole must remain a hole.
    core[17, 23] = True  # A disconnected island must remain separate.
    masks, offsets, alternatives = exact_controls(core, seed=99)
    again, same, count = exact_controls(core, seed=99)
    assert torch.equal(masks, again) and offsets == same and alternatives == count
    assert masks.shape == (5, 31, 47)
    for mask, shift in zip(masks, offsets):
        assert shift != (0, 0)
        assert torch.equal(core.nonzero() + torch.tensor(shift), mask.nonzero())
        assert mask.sum() == core.sum()


def test_no_legal_translation_and_limited_placements():
    with pytest.raises(NoLegalTranslation):
        exact_controls(torch.ones((3, 5), dtype=torch.bool), seed=0)
    with pytest.raises(NoLegalTranslation):
        exact_controls(torch.zeros((3, 5), dtype=torch.bool), seed=0)
    core = torch.zeros((2, 2), dtype=torch.bool)
    core[:, 0] = True
    masks, offsets, alternatives = exact_controls(core, seed=0)
    assert alternatives == 1 and offsets == [(0, 1)] * 5
    assert masks.sum().item() == 10  # With replacement, still five valid controls.


def test_mean_fill_preserves_every_unmasked_rgb_byte():
    rgb = torch.randint(0, 256, (3, 19, 23), generator=torch.Generator().manual_seed(9), dtype=torch.uint8)
    original = rgb.clone()
    core = torch.zeros((19, 23), dtype=torch.bool)
    core[3:8, 5:13] = True
    masks = torch.cat([core[None], exact_controls(core, seed=2)[0]])
    variants, mean, fill = native_mean_fill(rgb, masks)
    assert torch.equal(rgb, original)
    np.testing.assert_allclose(mean, (rgb.float()/255).mean((-2, -1)).numpy())
    for image, mask in zip(variants, masks):
        assert torch.equal(image[:, ~mask], rgb[:, ~mask])
        assert torch.equal(image[:, mask], torch.tensor(fill, dtype=torch.uint8)[:, None].expand(-1, int(mask.sum())))


class TinyBoQ(nn.Module):
    def forward(self, image):
        return F.normalize(F.adaptive_avg_pool2d(image, (4, 4)).flatten(1), dim=1), []


def test_full_evaluation_uses_direct_clean_forward_and_native_controls():
    model = TinyBoQ().eval().requires_grad_(False)
    rgb = torch.randint(0, 256, (3, 37, 59), generator=torch.Generator().manual_seed(2), dtype=torch.uint8)
    clean, _ = model(evaluation_transform()(rgb)[None])
    bank = RetrievalContext(torch.cat([clean, clean, -clean, -clean]), ["X:1", "X:1", "X:2", "X:2"])
    margin = bank.query(clean, "X:1")[0]["margin"]
    core = torch.zeros((37, 59), dtype=torch.bool)
    core[5:13, 8:21] = True
    rows, masks, audit = evaluate_core(model, bank, rgb, core, image_key="X/a.jpg", place_key="X:1", seed=0,
                                      cached_descriptor=clean, cached_margin=margin)
    assert len(rows) == 7 and masks.shape == (6, 37, 59)
    assert audit["descriptor_max_abs_error"] == audit["clean_margin_abs_error"] == 0
    variants, _, _ = native_mean_fill(rgb, torch.from_numpy(masks))
    descriptors, _ = model(evaluation_transform()(variants))
    direct = bank.query(descriptors, "X:1", clean_descriptor=clean)
    for row, value in zip(rows[1:], direct):
        assert row["margin_drop"] == margin-value["margin"]
    with pytest.raises(ValueError, match="Clean forward differs"):
        evaluate_core(model, bank, rgb, core, image_key="X/a.jpg", place_key="X:1", seed=0,
                      cached_descriptor=clean+0.1, cached_margin=margin)


def candidate(cid, weight, precision=.8, passed=True):
    return dict(candidate_id=cid, metrics=dict(vulnerability_weighted_coverage=weight,
                target_precision=precision, area_fraction=.06, centroid_distance_normalized=.1),
                thresholds={str(t): {"passes": passed if t == .3 else False} for t in (.3, .5, .7)})


def test_selection_ignores_response_and_is_order_independent():
    rows = [candidate("b", .35), candidate("a", .35), candidate("z", .9, passed=False)]
    rows[0]["margin_drop"] = 999  # Retrieval outcome cannot select a candidate.
    assert choose_core(rows, .3)["candidate_id"] == "a"
    assert choose_core(rows[::-1], .3)["candidate_id"] == "a"
    assert choose_core(rows, .5) is None
    with pytest.raises(ValueError):
        choose_core(rows, .4)


def condition_rows(key="a", place="X:1"):
    base = dict(arm=ARMS[0], image_key=key, place_key=place, candidate_id="c", family="parked_vehicle",
                paired_eligible=True, mask_pixels=42, mask_area_fraction=.06, clean_margin=.2)
    return [dict(base, condition=condition, random_repeat=repeat, margin_drop=drop, applied_mask_pixels=pixels)
            for condition, repeat, drop, pixels in [("clean", -1, 0., 0), ("targeted", -1, .12, 42)] +
            [("random", i, (i+1)*.01, 42) for i in range(5)]]


def test_random_repeats_collapse_within_query_and_incomplete_pairs_rejected():
    rows = condition_rows()
    pairs = pair_queries(rows)
    assert len(pairs) == 1 and pairs[0]["random_repeats"] == 5
    assert pairs[0]["random_mean_margin_drop"] == pytest.approx(.03)
    assert pairs[0]["paired_margin_difference"] == pytest.approx(.09)
    with pytest.raises(ValueError, match="exactly five"):
        pair_queries(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[-1]["applied_mask_pixels"] = 43
    with pytest.raises(ValueError, match="pixel count"):
        pair_queries(bad)


def test_query_bootstrap_20000_reproducible_empty_arms_and_freeze():
    stats = _module(Path(__file__).resolve().parents[1]/"scripts/analyze_stage1_results.py", "test_core_stats")
    pairs = pair_queries(condition_rows()+condition_rows("b", "X:2"))
    pairs[1]["target_margin_drop"] = .08
    a = summarize_pairs(pairs, ARMS[0], seed=0, stats_module=stats)
    assert a == summarize_pairs(pairs, ARMS[0], seed=0, stats_module=stats)
    assert a["n_queries"] == 2 and a["bootstrap_resamples"] == 20000 and a["bootstrap_unit"] == "query"
    assert a["mean_paired_difference"] == pytest.approx(.07)
    assert a["ci_low"] == pytest.approx(.05) and a["ci_high"] == pytest.approx(.09)
    assert a["win_rate"] == 1
    empty = summarize_pairs(pairs, ARMS[1], seed=0, stats_module=stats)
    assert empty["n_queries"] == 0 and empty["wilcoxon_p_raw"] is None
    assert freeze_choice([a, empty]) == a
    assert freeze_choice([empty]) is None
    high_coverage = dict(a, arm=ARMS[2], n_queries=3, mean_paired_difference=-.1, wilcoxon_p_raw=1.)
    assert freeze_choice([a, high_coverage]) == high_coverage
    with pytest.raises(ValueError, match="one query per place"):
        summarize_pairs(pairs+pairs, ARMS[0], seed=0, stats_module=stats)
