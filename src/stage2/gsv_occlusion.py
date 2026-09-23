"""Shape-matched GSV evaluation and query-paired statistics; no model changes."""

from collections import defaultdict
import hashlib

import numpy as np
import torch
from torchvision.transforms import v2 as T

from scripts.analyze_stage1_results import holm, paired_statistics
from src.analysis.masks import build_attention_mask, upsample_token_mask
from src.analysis.perturb import NoLegalTranslation, make_shape_matched_random_mask, perturb_rgb
from src.stage2.retrieval import MEAN, STD


ENDPOINTS = ("margin_drop", "descriptor_drift", "rank_degradation", "positive_similarity_drop",
             "failure_flip", "hit_at_1", "hit_at_5", "hit_at_10")


def stable_seed(seed, image_key, strategy):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{image_key}\0{strategy}".encode()).digest()[:8], "little") % (2**63)


def sample_unique_places(split, count, seed):
    """Uniform deterministic place ordering, then one hashed SOURCE per place."""
    groups = defaultdict(list)
    for record in split.source:
        groups[record.place_key].append(record)
    if count < 2 or count > len(groups):
        raise ValueError("Need 2..number_of_places distinct SOURCE places")
    places = sorted(groups, key=lambda key: (hashlib.sha256(f"place:{seed}\0{key}".encode()).digest(), key))[:count]
    return tuple(min(groups[key], key=lambda row: (hashlib.sha256(f"image:{seed}\0{row.image_key}".encode()).digest(),
                                                  row.image_key)) for key in places)


def shape_matched_controls(token_mask, image_size, repeats, seed):
    """Pixel translations of the exact target silhouette, with audited offsets."""
    target = upsample_token_mask(token_mask, image_size)
    masks, offsets, alternatives = make_shape_matched_random_mask(target, repeats, seed=seed)
    coordinates = target.nonzero()
    for mask, offset in zip(masks, offsets):
        if (offset == (0, 0) or not torch.equal(coordinates + coordinates.new_tensor(offset), mask.nonzero())
                or int(mask.sum()) != int(target.sum())):
            raise ValueError("Random mask is not an exact nonzero rigid translation")
    return target, masks, offsets, alternatives


@torch.no_grad()
def evaluate_query(model, rgb, record, bank, estimator, *, seed, repeats=5, batch_size=32):
    """Evaluate every preselected source, including weak/zero probe diagnostics.

    Only geometrically impossible translations exclude a source from a given
    strategy's paired analysis. The source remains in the raw results.
    """
    if repeats < 1 or batch_size < 1:
        raise ValueError("repeats and batch_size must be positive")
    if estimator.strict:
        raise ValueError("Evaluation requires strict=False to avoid outcome-dependent sample selection")
    output = estimator.estimate(model, rgb, record.place_key, bank)
    if max(output.diagnostics["extraction_descriptor_max_abs_error"],
           output.diagnostics["after_probe_descriptor_max_abs_error"]) > estimator.descriptor_tolerance:
        raise ValueError("STOP: clean descriptor drift invalidates evaluation")
    target_masks = {
        "attention": build_attention_mask(output.attention_map, estimator.mask_ratio),
        "fused": output.token_mask,
    }
    images, labels, geometry = [rgb[None]], [], {}
    for strategy, token_mask in target_masks.items():
        placement_seed = stable_seed(seed, record.image_key, strategy)
        target = upsample_token_mask(token_mask, rgb.shape[-2:])
        try:
            target, randoms, offsets, alternatives = shape_matched_controls(
                token_mask, rgb.shape[-2:], repeats, placement_seed)
            reason = ""
        except NoLegalTranslation as exc:
            randoms, offsets, alternatives, reason = target[None][:0], [], 0, str(exc)
        eligible = bool(alternatives)
        geometry[strategy] = dict(
            paired_eligible=eligible, exclusion_reason=reason, legal_random_locations=alternatives,
            unique_random_placements=len(set(offsets)), placement_seed=placement_seed,
            target_mask_tokens=int(token_mask.sum()), mask_pixels=int(target.sum()),
            actual_area_fraction=float(target.float().mean()),
        )
        masks = torch.cat([target[None], randoms])
        images.append(perturb_rgb(rgb, masks, operator="mean_fill"))
        labels.append((strategy, "targeted", -1, 0, 0))
        labels.extend((strategy, "random", repeat, dy, dx) for repeat, (dy, dx) in enumerate(offsets))
    normalize = T.Normalize(MEAN, STD)
    images = torch.cat(images)
    descriptors = torch.cat([model(normalize(batch))[0] for batch in images.split(batch_size)])
    error = float((descriptors[:1] - output.clean_descriptor).abs().max())
    if error > 2e-6:
        raise ValueError("STOP: batched clean descriptor disagrees with estimator")
    metrics = bank.query(descriptors, record.place_key, clean_descriptor=output.clean_descriptor)
    clean, rows = metrics[0], []

    def make_row(strategy, condition, repeat, dy, dx, metric):
        return dict(
            seed=seed, image_key=record.image_key, place_key=record.place_key, strategy=strategy,
            condition=condition, random_repeat=repeat, shift_y=dy, shift_x=dx,
            **geometry[strategy], clean_margin=clean["margin"], perturbed_margin=metric["margin"],
            clean_positive_sim=clean["positive_sim"], positive_sim=metric["positive_sim"],
            clean_negative_sim=clean["negative_sim"], negative_sim=metric["negative_sim"],
            clean_rank=clean["rank"], perturbed_rank=metric["rank"],
            margin_drop=clean["margin"] - metric["margin"],
            descriptor_drift=0. if condition == "clean" else metric["descriptor_drift"],
            positive_similarity_drop=clean["positive_sim"] - metric["positive_sim"],
            rank_degradation=metric["rank"] - clean["rank"],
            failure_flip=int(clean["hit_at_1"] and not metric["hit_at_1"]),
            hit_at_1=metric["hit_at_1"], hit_at_5=metric["hit_at_5"], hit_at_10=metric["hit_at_10"],
            applied_mask_pixels=0 if condition == "clean" else geometry[strategy]["mask_pixels"],
        )

    for strategy in target_masks:
        rows.append(make_row(strategy, "clean", -1, 0, 0, clean))
    for label, metric in zip(labels, metrics[1:]):
        rows.append(make_row(*label, metric))
    diagnostic = dict(seed=seed, image_key=record.image_key, place_key=record.place_key,
                      clean_batch_max_abs_error=error, **output.diagnostics)
    return rows, diagnostic, target_masks


def summarize_experiment(rows, *, seeds, repeats=5, resamples=20000, min_paired_fraction=.8):
    """Average random repeats first; resample paired queries, never placements.

    Attention and fused are separate hypothesis families (Holm across seeds).
    The attention family is the primary gate. Fused is supplementary.
    """
    if resamples < 10000 or repeats < 1 or not 0 < min_paired_fraction <= 1:
        raise ValueError("Require >=10,000 resamples, positive repeats and valid paired coverage")
    if not rows or len(set(seeds)) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
        raise ValueError("Rows must match distinct requested seeds")
    groups = defaultdict(list)
    for row in rows:
        if row["strategy"] not in ("attention", "fused"):
            raise ValueError("Unknown targeting strategy")
        groups[row["seed"], row["strategy"], row["image_key"]].append(row)
    pairs, exclusions = [], []
    identities = defaultdict(dict)
    for (seed, strategy, image_key), group in sorted(groups.items()):
        by_condition = defaultdict(list)
        for row in group:
            by_condition[row["condition"]].append(row)
            if any(not np.isfinite(row[key]) for key in ENDPOINTS):
                raise ValueError("Non-finite raw endpoints")
        if set(by_condition) - {"clean", "targeted", "random"}:
            raise ValueError("Unknown condition")
        if len(by_condition["clean"]) != 1 or len(by_condition["targeted"]) != 1:
            raise ValueError("Each query requires exactly one clean and targeted row")
        clean, target = by_condition["clean"][0], by_condition["targeted"][0]
        if len({r["place_key"] for r in group}) != 1 or len({r["paired_eligible"] for r in group}) != 1:
            raise ValueError("Mixed place identity or eligibility")
        identities[seed, strategy][image_key] = target["place_key"]
        if any(r["clean_margin"] != clean["clean_margin"] or
               not np.isclose(r["margin_drop"], r["clean_margin"] - r["perturbed_margin"], atol=1e-12, rtol=0)
               for r in group):
            raise ValueError("Margin accounting mismatch")
        if clean["margin_drop"] != 0 or clean["applied_mask_pixels"] != 0:
            raise ValueError("Invalid clean condition")
        randoms = by_condition["random"]
        if not target["paired_eligible"]:
            if randoms or not target["exclusion_reason"]:
                raise ValueError("Ineligible query must explain exclusion and have no random draws")
            exclusions.append(dict(seed=seed, strategy=strategy, image_key=image_key,
                                   place_key=target["place_key"], reason=target["exclusion_reason"]))
            continue
        if sorted(r["random_repeat"] for r in randoms) != list(range(repeats)):
            raise ValueError("Missing or duplicate random repeats")
        if any(r["applied_mask_pixels"] != target["applied_mask_pixels"] or (r["shift_y"], r["shift_x"]) == (0, 0)
               for r in randoms):
            raise ValueError("Unequal area or untranslated random mask")
        item = dict(seed=seed, strategy=strategy, image_key=image_key, place_key=target["place_key"],
                    clean_rank=clean["clean_rank"])
        for endpoint in ENDPOINTS:
            item[f"clean_{endpoint}"] = clean[endpoint]
            item[f"targeted_{endpoint}"] = target[endpoint]
            item[f"random_{endpoint}"] = float(np.mean([r[endpoint] for r in randoms]))
        item["paired_margin_difference"] = item["targeted_margin_drop"] - item["random_margin_drop"]
        pairs.append(item)
    for seed in seeds:
        if not identities[seed, "attention"] or identities[seed, "attention"] != identities[seed, "fused"]:
            raise ValueError("Both strategies must use exactly the same preselected source cohort")
        places = list(identities[seed, "attention"].values())
        if len(set(places)) != len(places):
            raise ValueError("Use only one SOURCE per place for query-level resampling")
    summaries, comparisons = [], []
    for strategy in ("attention", "fused"):
        for seed in seeds:
            group = [p for p in pairs if p["seed"] == seed and p["strategy"] == strategy]
            n_total = len(identities[seed, strategy])
            comparison = dict(seed=seed, strategy=strategy, n_total=n_total, n_paired=len(group),
                              n_excluded=n_total - len(group), paired_fraction=len(group)/n_total)
            if len(group) >= 2:
                result = paired_statistics([p["targeted_margin_drop"] for p in group],
                                           [p["random_margin_drop"] for p in group],
                                           seed=seed, resamples=resamples, key=f"gsv:{strategy}:margin")
                comparison.update({k: result[k] for k in ("mean_paired_difference", "ci_low", "ci_high",
                                                          "wilcoxon_p_raw", "left_win_rate", "rank_biserial")})
            else:
                comparison.update(mean_paired_difference=None, ci_low=None, ci_high=None,
                                  wilcoxon_p_raw=1., left_win_rate=None, rank_biserial=None)
            comparisons.append(comparison)
            for condition in ("clean", "random", "targeted"):
                for endpoint in ENDPOINTS:
                    values = [p[f"{condition}_{endpoint}"] for p in group]
                    summaries.append(dict(seed=seed, strategy=strategy, condition=condition, endpoint=endpoint,
                                          n_queries=len(values), mean=float(np.mean(values)) if values else None,
                                          variance=float(np.var(values, ddof=1)) if len(values) > 1 else None))
        family = [c for c in comparisons if c["strategy"] == strategy]
        for comparison, corrected in zip(family, holm([c["wilcoxon_p_raw"] for c in family])):
            comparison["wilcoxon_p_holm"] = float(corrected)
            comparison["supported"] = bool(comparison["ci_low"] is not None and comparison["ci_low"] > 0
                                            and comparison["mean_paired_difference"] > 0
                                            and corrected < .05 and comparison["paired_fraction"] >= min_paired_fraction)
    primary = [c for c in comparisons if c["strategy"] == "attention"]
    return dict(status="GO" if all(c["supported"] for c in primary) else "STOP",
                primary="attention targeted minus own shape-matched random mean",
                reference_seed_consistency_checked=len(seeds) >= 2,
                random_repeats=repeats, bootstrap_resamples=resamples, bootstrap_unit="one SOURCE per place",
                wilcoxon_alternative="two-sided", correction="Holm across seeds within each strategy",
                min_paired_fraction=min_paired_fraction, comparisons=comparisons,
                summaries=summaries, pairs=pairs, exclusions=exclusions)
