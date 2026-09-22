"""Exact-budget token masks and nearest-neighbor pixel expansion."""

import heapq
import math

import torch
import torch.nn.functional as F


def build_attention_mask(token_map, ratio, mode="connected_topk"):
    """Return a bool mask with True indicating selected tokens.

    Input/output: [Ht, Wt] or [B, Ht, Wt]. Use raw attention scores to avoid
    percentile clipping ties. Each image selects exactly round(ratio * Ht * Wt)
    tokens (Python round, ties to even). Supports any ratio in [0, 1], including
    0.10, 0.15 and 0.20. No dilation or pixel-level re-ranking is performed.

    raw_topk selects the globally largest scores. connected_topk starts at the
    maximum and repeatedly selects the highest-scoring 8-neighbor frontier
    token. Ties in both modes use the smallest row-major index, deterministically.
    """
    if mode not in {"raw_topk", "connected_topk"}:
        raise ValueError("mode must be raw_topk or connected_topk")
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be finite and in [0, 1]")
    if token_map.ndim not in {2, 3} or any(size == 0 for size in token_map.shape):
        raise ValueError("token_map must be nonempty [Ht, Wt] or [B, Ht, Wt]")
    if not torch.isfinite(token_map).all():
        raise ValueError("token_map must contain finite values")
    squeeze = token_map.ndim == 2
    maps = token_map.unsqueeze(0) if squeeze else token_map
    batch_size, height, width = maps.shape
    num_tokens = height * width
    budget = round(float(ratio) * num_tokens)
    scores = maps.detach().cpu().reshape(batch_size, num_tokens)
    masks = torch.zeros((batch_size, num_tokens), dtype=torch.bool)
    for b, values in enumerate(scores):
        if budget == 0:
            continue
        if mode == "raw_topk":
            indices = torch.argsort(values, descending=True, stable=True)[:budget]
            masks[b, indices] = True
            continue
        start = int(values.argmax())
        frontier = [(-float(values[start]), start)]
        enqueued = {start}
        for _ in range(budget):
            _, index = heapq.heappop(frontier)
            masks[b, index] = True
            row, col = divmod(index, width)
            for r in range(max(0, row - 1), min(height, row + 2)):
                for c in range(max(0, col - 1), min(width, col + 2)):
                    neighbor = r * width + c
                    if neighbor not in enqueued:
                        enqueued.add(neighbor)
                        heapq.heappush(frontier, (-float(values[neighbor]), neighbor))
    assert (masks.sum(dim=1) == budget).all()
    masks = masks.reshape(batch_size, height, width).to(token_map.device)
    return masks[0] if squeeze else masks


def upsample_token_mask(token_mask, image_size):
    """Expand bool [Ht,Wt]/[B,Ht,Wt] masks to (H,W) with nearest only.

    For image dimensions not divisible by the grid, token cells can occupy
    unequal pixel areas. The token budget remains exact; report pixel area too.
    """
    if token_mask.ndim not in {2, 3} or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean [Ht,Wt] or [B,Ht,Wt]")
    if any(size == 0 for size in token_mask.shape):
        raise ValueError("token_mask must be nonempty")
    if len(image_size) != 2 or any(int(s) != s or s <= 0 for s in image_size):
        raise ValueError("image_size must contain two positive integers")
    squeeze = token_mask.ndim == 2
    masks = token_mask.unsqueeze(0) if squeeze else token_mask
    pixels = F.interpolate(masks[:, None].float(), size=tuple(map(int, image_size)), mode="nearest")
    pixels = pixels[:, 0].bool()
    return pixels[0] if squeeze else pixels
