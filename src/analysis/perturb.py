"""RGB-space occlusion and exact shape-matched random translations."""

import torch
from torchvision.transforms.functional import gaussian_blur


class NoLegalTranslation(ValueError):
    """The mask cannot move to another in-bounds location without deformation."""


def make_shape_matched_random_mask(attention_mask, repeats=5, *, seed):
    """Translate a bool [H,W] pixel mask; never crop, wrap, rotate or dilate.

    Sample uniformly WITH replacement over all legal integer-pixel translations,
    excluding the original location. Overlap with the original mask is allowed.
    Returns masks [repeats,H,W], (dy,dx) offsets and number of legal alternatives.
    If none exist, raise NoLegalTranslation; the evaluator must report exclusion
    from paired analysis, never substitute a different-shaped random mask.
    """
    if attention_mask.ndim != 2 or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean [H,W]")
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError("repeats must be a positive integer")
    coordinates = attention_mask.nonzero()
    if coordinates.numel() == 0:
        raise NoLegalTranslation("Empty mask has no distinct translated support")
    top, left = coordinates.min(0).values.tolist()
    bottom, right = coordinates.max(0).values.tolist()
    height, width = attention_mask.shape
    crop = attention_mask[top:bottom + 1, left:right + 1]
    ny, nx = height - crop.shape[0] + 1, width - crop.shape[1] + 1
    alternatives = ny * nx - 1
    if alternatives == 0:
        raise NoLegalTranslation("Mask bounding box spans the full image")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    original = top * nx + left
    draws = torch.randint(alternatives, (repeats,), generator=generator)
    draws += (draws >= original).long()  # Skip the original placement exactly.
    masks = torch.zeros((repeats, height, width), dtype=torch.bool, device=attention_mask.device)
    offsets = []
    for index, draw in enumerate(draws.tolist()):
        row, col = divmod(draw, nx)
        masks[index, row:row + crop.shape[0], col:col + crop.shape[1]] = crop
        offsets.append((row - top, col - left))
    assert (masks.sum((1, 2)) == attention_mask.sum()).all()
    assert all(offset != (0, 0) for offset in offsets)
    return masks, offsets, alternatives


def make_token_aligned_random_mask(token_mask, repeats=5, *, seed):
    """Translate the ORIGINAL bool [Ht,Wt] mask before any upsampling.

    Returns token masks [repeats,Ht,Wt], token offsets (dy,dx), and the number
    of legal alternative locations. Excludes the original location. Samples
    uniformly without replacement when alternatives >= repeats; otherwise uses
    every alternative once in random order, then samples the remainder with
    replacement. A pure translation preserves token count and topology exactly.

    Apply nearest upsampling to the returned masks. When RGB dimensions are
    integer multiples of the grid, pixel shape and area are also exactly equal.
    Nonuniform nearest cells need an explicit area check by the caller.
    """
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean [Ht,Wt]")
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError("repeats must be a positive integer")
    coordinates = token_mask.nonzero()
    if coordinates.numel() == 0:
        raise NoLegalTranslation("Empty token mask has no translated support")
    top, left = coordinates.min(0).values.tolist()
    bottom, right = coordinates.max(0).values.tolist()
    height, width = token_mask.shape
    crop = token_mask[top:bottom + 1, left:right + 1]
    ny, nx = height - crop.shape[0] + 1, width - crop.shape[1] + 1
    alternatives = ny * nx - 1
    if alternatives == 0:
        raise NoLegalTranslation("Token mask bounding box spans the full grid")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = torch.randperm(alternatives, generator=generator)[:repeats]
    if alternatives < repeats:
        draws = torch.cat([draws, torch.randint(alternatives, (repeats - alternatives,), generator=generator)])
    original = top * nx + left
    draws += (draws >= original).long()
    masks = torch.zeros((repeats, height, width), dtype=torch.bool, device=token_mask.device)
    offsets = []
    for index, draw in enumerate(draws.tolist()):
        row, col = divmod(draw, nx)
        masks[index, row:row + crop.shape[0], col:col + crop.shape[1]] = crop
        offsets.append((row - top, col - left))
    assert (masks.sum((1, 2)) == token_mask.sum()).all()
    assert (0, 0) not in offsets
    assert len(set(offsets)) == min(repeats, alternatives)
    return masks, offsets, alternatives

# 对 RGB 图像应用掩码，实现遮挡或模糊效果
# operator 表示操作符，mean_fill 表示用均值填充，gaussian_blur 表示应用高斯模糊
# fill_rgb 表示填充 RGB 值，blur_kernel 表示模糊核大小，blur_sigma 表示模糊标准差
# 返回 [B,3,H,W] 形状的图像
def perturb_rgb(rgb, masks, *, operator="mean_fill", fill_rgb=None,
                blur_kernel=31, blur_sigma=5.0):
    """Apply masks to RGB float [0,1], BEFORE ImageNet normalization.

    rgb: [3,H,W] or [B,3,H,W]; masks: [H,W] or [B,H,W] bool.
    A single RGB image broadcasts to multiple masks. Default mean fill is the
    original image's per-channel spatial mean, shared across all placements.
    An explicit RGB triple (e.g. ImageNet mean) can be supplied instead.
    gaussian_blur is an optional secondary operator with the same mask support.
    Always returns [B,3,H,W], without mutating the original image.
    """
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
        raise ValueError("rgb must be floating [3,H,W] or [B,3,H,W]")
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Occlusion requires finite RGB [0,1], not normalized tensors")
    if (masks.ndim != 3 or masks.dtype != torch.bool or masks.shape[-2:] != rgb.shape[-2:]
            or rgb.shape[0] not in {1, masks.shape[0]} or masks.device != rgb.device):
        raise ValueError("Masks must match RGB spatial size, batch and device")
    if operator == "mean_fill":
        if fill_rgb is None:
            fill = rgb.mean(dim=(-2, -1), keepdim=True)
        else:
            fill = torch.as_tensor(fill_rgb, dtype=rgb.dtype, device=rgb.device)
            if fill.shape != (3,) or not torch.isfinite(fill).all() or fill.min() < 0 or fill.max() > 1:
                raise ValueError("fill_rgb must be a finite RGB [0,1] triple")
            fill = fill[None, :, None, None]
    elif operator == "gaussian_blur":
        if blur_kernel < 1 or blur_kernel % 2 != 1 or blur_sigma <= 0:
            raise ValueError("Gaussian kernel must be positive odd, sigma positive")
        fill = gaussian_blur(rgb, [blur_kernel, blur_kernel], [blur_sigma, blur_sigma])
    else:
        raise ValueError("operator must be mean_fill or gaussian_blur")
    return torch.where(masks[:, None], fill, rgb)


def mask_iou(mask, attention_mask):
    union = (mask | attention_mask).sum().item()
    return (mask & attention_mask).sum().item() / union if union else 1.0
