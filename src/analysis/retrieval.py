"""Exact full-database retrieval metrics and query-level paired summaries."""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def retrieval_metrics(descriptors, references, positive_indices, clean_descriptor):
    """Metrics for multiple conditions of ONE query, against the full database.

    Descriptors must be L2-normalized. Ground truth indices refer to references.
    Rank is 1-based; exact similarity ties are ordered by reference index.
    A negative means any database image outside this query's ground-truth set.
    """
    if descriptors.ndim != 2 or references.ndim != 2 or descriptors.shape[1] != references.shape[1]:
        raise ValueError("Descriptors/references must be compatible [N,D] tensors")
    positives = torch.as_tensor(positive_indices, device=references.device, dtype=torch.long).unique()
    nref = references.shape[0]
    if positives.numel() == 0 or positives.min() < 0 or positives.max() >= nref:
        raise ValueError("Ground truth must contain valid reference indices")
    if positives.numel() == nref:
        raise ValueError("Hardest negative requires at least one non-positive reference")
    similarities = descriptors @ references.T
    if not torch.isfinite(similarities).all():
        raise ValueError("Non-finite similarities")
    positive_scores = similarities[:, positives]
    best_positive, position = positive_scores.max(dim=1)
    best_positive_id = positives[position]  # unique() returns sorted indices.
    negatives = torch.ones(nref, dtype=torch.bool, device=references.device)
    negatives[positives] = False
    best_negative = similarities[:, negatives].max(dim=1).values
    ids = torch.arange(nref, device=references.device)
    ranks = 1 + (similarities > best_positive[:, None]).sum(1)
    ranks += ((similarities == best_positive[:, None]) & (ids[None] < best_positive_id[:, None])).sum(1)
    drift = (1 - F.cosine_similarity(descriptors, clean_descriptor.reshape(1, -1))).clamp(0, 2)
    results = []
    for i in range(len(descriptors)):
        rank = int(ranks[i])
        results.append({
            "rank": rank, "positive_sim": float(best_positive[i]),
            "negative_sim": float(best_negative[i]),
            "margin": float(best_positive[i] - best_negative[i]),
            "descriptor_drift": float(drift[i]),
            "hit_at_1": int(rank <= 1), "hit_at_5": int(rank <= 5),
            "hit_at_10": int(rank <= 10),
        })
    return results


ENDPOINTS = ["margin_drop", "descriptor_drift", "positive_similarity_drop",
             "rank_degradation", "failure_flip", "hit_at_1", "hit_at_5", "hit_at_10"]


def paired_query_rows(rows, ratio, repeats=5):
    """Average random draws WITHIN query; one independent row per query.

    Queries with no legal translations are excluded from ALL conditions in
    the paired comparison, with exclusion counts reported separately.
    """
    targets = {r["query_id"]: r for r in rows if r["condition"] == "attention" and r["mask_ratio"] == ratio}
    randoms = {}
    for row in rows:
        if row["condition"] == "random" and row["mask_ratio"] == ratio:
            randoms.setdefault(row["query_id"], []).append(row)
    paired = []
    for query_id, target in targets.items():
        draws = randoms.get(query_id, [])
        if not target["paired_eligible"]:
            if draws:
                raise ValueError("Ineligible query unexpectedly has random rows")
            continue
        if len(draws) != repeats or sorted(r["random_repeat"] for r in draws) != list(range(repeats)):
            raise ValueError("Each eligible query requires exactly the configured random repeats")
        item = {"query_id": query_id, "mask_ratio": ratio, "clean_rank": target["clean_rank"]}
        for key in ENDPOINTS:
            item[f"attention_{key}"] = target[key]
            item[f"random_{key}"] = float(np.mean([r[key] for r in draws]))
        item["paired_margin_difference"] = item["attention_margin_drop"] - item["random_margin_drop"]
        paired.append(item)
    return paired


def summarize(rows, ratios, *, repeats=5, seed=2024, bootstrap_samples=2000):
    """Descriptive mean effects and query-bootstrap 95% CIs (no draw pooling)."""
    summaries, all_pairs = [], []
    for ratio in ratios:
        pairs = paired_query_rows(rows, ratio, repeats)
        all_pairs.extend(pairs)
        total = sum(r["condition"] == "attention" and r["mask_ratio"] == ratio for r in rows)
        base = {"mask_ratio": ratio, "n_queries_total": total, "n_paired_queries": len(pairs),
                "n_excluded_no_translation": total - len(pairs)}
        for condition in ["clean", "random", "attention"]:
            item = dict(base, condition=condition)
            for key in ENDPOINTS:
                if condition == "clean":
                    values = [int(p["clean_rank"] <= int(key.rsplit("_", 1)[1])) if key.startswith("hit_at_") else 0.0 for p in pairs]
                else:
                    values = [p[f"{condition}_{key}"] for p in pairs]
                item[key] = float(np.mean(values)) if values else None
            # Unconditional flips / all paired queries and conditional / clean successes.
            clean_hits = sum(p["clean_rank"] == 1 for p in pairs)
            item["n_clean_correct"] = clean_hits
            item["failure_flip_given_clean_correct"] = (
                sum(p[f"{condition}_failure_flip"] for p in pairs) / clean_hits
                if clean_hits and condition != "clean" else (0.0 if clean_hits else None))
            summaries.append(item)
        differences = np.array([p["paired_margin_difference"] for p in pairs])
        ci = [None, None]
        if len(differences) and bootstrap_samples:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(ratio * 1e6))]))
            means = [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(bootstrap_samples)]
            ci = np.quantile(means, [.025, .975]).tolist()
        for item in summaries[-3:]:
            item.update(paired_margin_difference=float(differences.mean()) if len(differences) else None,
                        paired_margin_ci_low=ci[0], paired_margin_ci_high=ci[1])
    return summaries, all_pairs
