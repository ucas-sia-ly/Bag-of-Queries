"""Attention Proposal Maps; these maps do not measure causal effects."""

import torch
import torch.nn.functional as F

# 实现(x-min)/(max-min)，使得映射后的值在 5% 到 95% 之间，注意力地图更容易可视化
def robust_normalize(maps, lower_percentile=5.0, upper_percentile=95.0):
    """Normalize each [H, W] image independently in a [B, H, W] tensor.

    Values outside the percentile interval are clipped. A collapsed interval
    produces zeros (including constant maps); raw scores remain available for
    mask ranking. Quantiles are calculated in float32.
    """
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Require 0 <= lower_percentile < upper_percentile <= 100")
    if maps.ndim != 3 or any(size == 0 for size in maps.shape):
        raise ValueError("maps must be nonempty [B, H, W]")
    if not torch.isfinite(maps).all():
        raise ValueError("maps must contain finite values")
    maps = maps.float()
    quantiles = maps.new_tensor([lower_percentile, upper_percentile]) / 100
    low, high = torch.quantile(maps.flatten(1), quantiles, dim=1)
    low, high = low[:, None, None], high[:, None, None]
    span = high - low
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    normalized = ((maps - low) / denominator).clamp(0, 1)
    return torch.where(span > 0, normalized, torch.zeros_like(normalized))

# 聚合注意力权重，求注意力地图的平均值
def aggregate_attention(attentions, token_h, token_w):
    """Mean heads -> queries -> layers, returning raw [B, Ht, Wt].

    Accept each layer as [B, Heads, Queries, Tokens] or already head-averaged
    [B, Queries, Tokens]. Spatial dimensions must come from backbone output.
    """
    if not attentions or token_h <= 0 or token_w <= 0:
        raise ValueError("Require attention layers and a nonempty token grid")
    batch_size = attentions[0].shape[0]
    layer_maps = []
    for attn in attentions:
        assert attn.ndim in {3, 4}
        assert attn.shape[0] == batch_size
        assert attn.shape[-1] == token_h * token_w
        if any(size == 0 for size in attn.shape) or not torch.isfinite(attn).all():
            raise ValueError("Attention must be nonempty and finite")
        attn = attn.detach().float()
        if attn.ndim == 4:
            attn = attn.mean(dim=1)
        layer_maps.append(attn.mean(dim=1))
    return torch.stack(layer_maps).mean(dim=0).reshape(batch_size, token_h, token_w)


@torch.no_grad()
# 提取 BoQ 模型的注意力地图
def extract_attention_map(
    model,
    normalized_images,
    *,
    return_head_attn=True,
    return_attentions=False,
    lower_percentile=5.0,
    upper_percentile=95.0,
):
    """Extract maps from a BoQ model with ``backbone`` and ``aggregator``.

    Input: ImageNet-normalized [B, 3, H, W], on the model's device.
    Output dictionary:
      descriptor: [B, descriptor_dim], unchanged BoQ descriptor.
      token_attention_map: raw aggregated [B, Ht, Wt], for mask ranking.
      normalized_token_attention_map: per-image robust [0, 1] display map.
      pixel_attention_map: bilinear display map [B, H, W].
      token_grid: (Ht, Wt), read directly from the backbone output.
      attentions: optional tuple of detached, per-layer attention tensors.

    Runs in eval mode without gradients and restores all module training flags.
    No image masking, second model pass, or retrieval evaluation is performed.
    """
    if (normalized_images.ndim != 4 or normalized_images.shape[1] != 3
            or any(size == 0 for size in normalized_images.shape)
            or not normalized_images.is_floating_point()):
        raise ValueError("normalized_images must be nonempty floating [B, 3, H, W]")
    if not torch.isfinite(normalized_images).all():
        raise ValueError("normalized_images must be finite")
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        features = model.backbone(normalized_images)
        assert features.ndim == 4
        assert features.shape[0] == normalized_images.shape[0]
        token_h, token_w = features.shape[-2:]
        descriptor, attentions = model.aggregator(
            features, return_head_attn=return_head_attn
        )
        token_map = aggregate_attention(attentions, token_h, token_w)
        display_map = robust_normalize(token_map, lower_percentile, upper_percentile)
        pixel_map = F.interpolate(
            display_map[:, None], size=normalized_images.shape[-2:],
            mode="bilinear", align_corners=False,
        )[:, 0].clamp(0, 1)
        result = {
            "descriptor": descriptor.detach(),
            "token_attention_map": token_map,
            "normalized_token_attention_map": display_map,
            "pixel_attention_map": pixel_map,
            "token_grid": (token_h, token_w),
        }
        if return_attentions:
            result["attentions"] = tuple(attn.detach() for attn in attentions)
        return result
    finally:
        for module, training in training_states:
            module.training = training
