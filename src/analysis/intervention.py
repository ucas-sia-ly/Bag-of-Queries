"""Local deletion audit of an Attention Proposal Map, not a causal map.

Window outcomes are dependent within image. Only query summaries are used for
bootstrap inference. Negative intervention damage is retained throughout.
"""

import numpy as np
from scipy import stats
import torch

from .attention import extract_attention_map
from .masks import upsample_token_mask
from .perturb import perturb_rgb
from .retrieval import retrieval_metrics


def token_windows(token_grid, window_size=2, stride=1, *, device="cpu"):
    """Return bool [W,Ht,Wt] masks and row-major [W,2] top-left positions."""
    h, w = token_grid
    if any(int(v) != v or v < 1 for v in (h, w, window_size, stride)):
        raise ValueError("Grid, window and stride must be positive integers")
    if min(h, w) < window_size:
        raise ValueError("Window does not fit the actual backbone grid")
    y, x = torch.meshgrid(torch.arange(0, h-window_size+1, stride, device=device),
                          torch.arange(0, w-window_size+1, stride, device=device), indexing="ij")
    positions = torch.stack([y.flatten(), x.flatten()], dim=1)
    yy = torch.arange(h, device=device)[None, :, None]
    xx = torch.arange(w, device=device)[None, None, :]
    masks = ((yy >= positions[:, 0, None, None]) & (yy < positions[:, 0, None, None]+window_size)
             & (xx >= positions[:, 1, None, None]) & (xx < positions[:, 1, None, None]+window_size))
    assert masks.shape == (len(positions), h, w)
    assert (masks.sum((1, 2)) == window_size**2).all()
    return masks, positions


@torch.no_grad()
def probe_query(model, rgb, normalize, references, positives, query_id, *,
                batch_size=32, window_size=2, stride=1, fill_rgb=None):
    """Batch deletion variants, preserving full-database retrieval semantics.

RGB must be [3,H,W] in [0,1]. Clean attention uses the existing head-averaged
path followed by mean queries / layers. Pixel masks are allocated per batch.
The final partial batch is allowed; there is no per-window model-forward loop.
"""
    if batch_size < 2 or int(batch_size) != batch_size:
        raise ValueError("Use batch_size >= 2 for intervention variants")
    if model.training:
        raise ValueError("Intervention probe requires an eval-mode frozen model")
    if (rgb.ndim != 3 or rgb.shape[0] != 3 or not rgb.is_floating_point()
            or not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1):
        raise ValueError("Probe input must be finite RGB [3,H,W] in [0,1]")
    result = extract_attention_map(model, normalize(rgb).unsqueeze(0), return_head_attn=False)
    raw = result["token_attention_map"][0]
    h, w = result["token_grid"]
    if rgb.shape[-2] % h or rgb.shape[-1] % w:
        raise ValueError("Equal pixel-area windows require RGB dimensions divisible by actual token grid")
    masks, positions = token_windows((h, w), window_size, stride, device=rgb.device)
    attention = (masks * raw[None]).sum((1, 2)) / window_size**2
    descriptor = result["descriptor"]
    clean = retrieval_metrics(descriptor, references, positives, descriptor)[0]
    rows, calls, batch_sizes = [], 0, []
    for first in range(0, len(masks), batch_size):
        pixels = upsample_token_mask(masks[first:first+batch_size], rgb.shape[-2:])
        assert (pixels.sum((1, 2)) == window_size**2 * (rgb.shape[-2]//h) * (rgb.shape[-1]//w)).all()
        variants = perturb_rgb(rgb, pixels, operator="mean_fill", fill_rgb=fill_rgb)
        values, _ = model(normalize(variants))
        metrics = retrieval_metrics(values, references, positives, descriptor)
        calls += 1
        batch_sizes.append(len(values))
        for index, metric in enumerate(metrics, start=first):
            y, x = positions[index].tolist()
            rows.append(dict(query_id=int(query_id), window_y=y, window_x=x,
                             attention_score=float(attention[index]),
                             margin_drop=clean["margin"]-metric["margin"],
                             positive_similarity_drop=clean["positive_sim"]-metric["positive_sim"],
                             descriptor_drift=metric["descriptor_drift"],
                             rank_degradation=metric["rank"]-clean["rank"],
                             clean_hit_at_1=clean["hit_at_1"], clean_rank=clean["rank"],
                             perturbed_rank=metric["rank"], clean_margin=clean["margin"],
                             perturbed_margin=metric["margin"], mask_tokens=window_size**2,
                             mask_pixels=window_size**2 * (rgb.shape[-2]//h) * (rgb.shape[-1]//w)))
    return dict(rows=rows, clean=clean, token_attention_map=raw.cpu().numpy(),
                token_grid=(h, w), positions=positions.cpu().numpy(),
                variant_forward_calls=calls, variant_batch_sizes=batch_sizes)


def summarize_query(rows, tail_fraction=.10):
    """One independent summary per query. Damage percentile: 100 = highest.

Top/bottom K use raw attention with row-major tie breaking. Damage percentiles
use average ranks: 100*(rank-1)/(N-1). A constant score produces undefined rho,
reported explicitly rather than coerced to zero. No window-level p-values.
"""
    if not 0 < tail_fraction <= .5 or len(rows) < 2:
        raise ValueError("Need >=2 windows and tail_fraction in (0,.5]")
    if len({r["query_id"] for r in rows}) != 1:
        raise ValueError("Summarize one query at a time")
    rows = sorted(rows, key=lambda r: (r["window_y"], r["window_x"]))
    attention = np.array([r["attention_score"] for r in rows])
    damage = np.array([r["margin_drop"] for r in rows])
    if not np.isfinite(attention).all() or not np.isfinite(damage).all():
        raise ValueError("Non-finite window values")
    valid = np.ptp(attention) > 0 and np.ptp(damage) > 0
    rho = float(stats.spearmanr(attention, damage).statistic) if valid else None
    k = max(1, round(tail_fraction * len(rows)))
    order = np.argsort(-attention, kind="stable")
    top, bottom = order[:k], order[-k:]  # Disjoint even for constant / tied attention.
    maximum = int(np.argmax(attention))
    percentile = 100*(stats.rankdata(damage, method="average")-1)/(len(damage)-1)
    return dict(query_id=rows[0]["query_id"], n_windows=len(rows), spearman=rho,
                spearman_defined=bool(valid), undefined_reason="" if valid else "constant attention or damage",
                tail_k=k, top_mean_damage=float(damage[top].mean()), bottom_mean_damage=float(damage[bottom].mean()),
                top_minus_bottom_damage=float(damage[top].mean()-damage[bottom].mean()),
                attention_max_window_y=rows[maximum]["window_y"], attention_max_window_x=rows[maximum]["window_x"],
                attention_max_ties=int((attention == attention.max()).sum()),
                attention_max_damage=float(damage[maximum]),
                attention_max_damage_percentile=float(percentile[maximum]),
                negative_damage_window_fraction=float((damage < 0).mean()),
                clean_hit_at_1=rows[0]["clean_hit_at_1"])


def bootstrap_summary(values, *, seed, resamples=20000):
    """Query resampling for mean, median and positive fraction, shared IDs."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or resamples < 1:
        raise ValueError("Invalid query bootstrap data")
    rng = np.random.default_rng(seed)
    means, medians, positive = [np.empty(resamples) for _ in range(3)]
    for start in range(0, resamples, 256):
        samples = x[rng.integers(0, len(x), (min(256, resamples-start), len(x)))]
        means[start:start+len(samples)] = samples.mean(1)
        medians[start:start+len(samples)] = np.median(samples, axis=1)
        positive[start:start+len(samples)] = (samples > 0).mean(1)
    result = dict(n_queries=len(x))
    for name, point, boot in [("mean", x.mean(), means), ("median", np.median(x), medians),
                              ("positive_fraction", (x > 0).mean(), positive)]:
        lo, hi = np.quantile(boot, [.025, .975])
        result.update({name: float(point), name+"_ci_low": float(lo), name+"_ci_high": float(hi)})
    return result


def summarize_faithfulness(queries, *, seed=2024, resamples=20000, min_mean_spearman=.30):
    """Descriptive strength threshold, not a preregistered test or calibration.

Stable positive association is recorded separately from sufficient local
faithfulness; significance alone must not turn a weak effect into a strong one.
"""
    ids = [r["query_id"] for r in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("Query summaries must be unique; windows are not independent samples")
    rho = [r["spearman"] for r in queries if r["spearman"] is not None]
    result = dict(n_queries=len(queries), n_spearman_defined=len(rho), n_spearman_undefined=len(queries)-len(rho),
                  bootstrap_unit="query", bootstrap_resamples=resamples, seed=seed,
                  min_mean_spearman=min_mean_spearman)
    result["spearman"] = bootstrap_summary(rho, seed=seed, resamples=resamples) if rho else None
    for index, field in enumerate(["top_minus_bottom_damage", "attention_max_damage_percentile"]):
        result[field] = bootstrap_summary([r[field] for r in queries], seed=seed+index+1, resamples=resamples)
    signal = (len(rho) == len(queries) and result["spearman"]["mean_ci_low"] > 0
              and result["top_minus_bottom_damage"]["mean_ci_low"] > 0)
    result["positive_proposal_signal_supported"] = bool(signal)
    useful = signal and result["spearman"]["mean"] >= min_mean_spearman
    result["conclusion"] = ("attention is a useful proposal signal" if useful else
                            "attention is useful for targeted occlusion globally, but is not sufficiently faithful as a local vulnerability estimator")
    return result


def project_window_values(positions, values, token_grid, window_size=2):
    """DISPLAY ONLY: signed mean over covering windows, with overlap counts.

This is not a per-token effect estimate. Uncovered tokens remain NaN.
"""
    sums, counts = np.zeros(token_grid), np.zeros(token_grid, dtype=int)
    for (y, x), value in zip(positions, values):
        sums[y:y+window_size, x:x+window_size] += value
        counts[y:y+window_size, x:x+window_size] += 1
    return np.divide(sums, counts, out=np.full(token_grid, np.nan), where=counts > 0), counts
