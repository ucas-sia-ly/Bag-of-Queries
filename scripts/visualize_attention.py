"""Visualize Attention Proposal Maps and exact-budget masks, without probing.

Example (run from the repository root):
    python scripts/visualize_attention.py --checkpoint CHECKPOINT \
        --backbone dinov2_vitb14 --image IMAGE --image-size 322 322 \
        --output-dir OUTPUT --seed 2024

Both raw state dicts (.pth) and Lightning checkpoints (.ckpt) are supported.
BoQ dimensions are read from checkpoint tensors, then loaded strictly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
from torchvision.transforms import v2 as T

from hubconf import VPRModel
from src.analysis import build_attention_mask, extract_attention_map, upsample_token_mask
from src.backbones import DinoV2, ResNet
from src.boq import BoQ


def load_model(checkpoint_path, backbone_name, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    proj_channels, in_channels = state["aggregator.proj_c.weight"].shape[:2]
    num_queries = state["aggregator.boqs.0.queries"].shape[1]
    layer_ids = sorted({int(key.split(".")[2]) for key in state if key.startswith("aggregator.boqs.")})
    if layer_ids != list(range(len(layer_ids))):
        raise ValueError("Checkpoint BoQ layers must be contiguous, starting from zero")
    row_dim, fc_inputs = state["aggregator.fc.weight"].shape
    if fc_inputs != len(layer_ids) * num_queries:
        raise ValueError("Checkpoint Query count and FC shape disagree")
    if backbone_name in DinoV2.AVAILABLE_MODELS:
        backbone = DinoV2(backbone_name=backbone_name)
    else:
        # The full checkpoint supplies all weights, so no ImageNet download.
        backbone = ResNet(backbone_name=backbone_name, pretrained=False, unfreeze_n_blocks=0)
    if backbone.out_channels != in_channels:
        raise ValueError("Backbone output channels do not match the checkpoint")
    aggregator = BoQ(in_channels, proj_channels, num_queries, len(layer_ids), row_dim)
    model = VPRModel(backbone, aggregator)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    config = {
        "backbone": backbone_name, "in_channels": in_channels,
        "proj_channels": proj_channels, "num_queries": num_queries,
        "num_layers": len(layer_ids), "row_dim": row_dim,
        "heads": aggregator.boqs[0].cross_attn.num_heads,
        "descriptor_dim": proj_channels * row_dim,
    }
    return model.to(device), config


def to_pil(rgb):
    """Convert an RGB [0,1] tensor to a display image."""
    array = rgb.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def colorize(values):
    # Fixed blue -> cyan -> yellow -> red scale for normalized [0,1] maps.
    anchors = values.new_tensor([[0.05, 0.05, 0.35], [0, 0.65, 0.9], [1, 0.9, 0], [0.85, 0, 0]])
    position = values.clamp(0, 1) * (len(anchors) - 1)
    index = position.long().clamp(max=len(anchors) - 2)
    fraction = (position - index)[..., None]
    return (anchors[index] * (1 - fraction) + anchors[index + 1] * fraction).permute(2, 0, 1)


def overlay_mask(rgb, mask, alpha):
    # All blending occurs on RGB [0,1], never on ImageNet-normalized tensors.
    red = rgb.new_tensor([1.0, 0.1, 0.1])[:, None, None]
    return torch.where(mask[None], (1 - alpha) * rgb + alpha * red, rgb)


def save_montage(panels, path, image_size):
    height, width = image_size
    margin, label_height, title_height = 12, 44, 48
    rows = (len(panels) + 3) // 4
    canvas = Image.new("RGB", (4 * (width + margin) + margin,
                               rows * (height + label_height + margin) + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((margin, 10), "Attention Proposal Map | blue: low, red: high | masks: nearest", fill="black", font=font)
    for index, (panel, label) in enumerate(panels):
        row, col = divmod(index, 4)
        left = margin + col * (width + margin)
        top = title_height + row * (height + label_height + margin)
        canvas.paste(to_pil(panel), (left, top))
        draw.multiline_text((left, top + height + 4), label, fill="black", font=font, spacing=2)
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", choices=DinoV2.AVAILABLE_MODELS + list(ResNet.AVAILABLE_MODELS), required=True)
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lower-percentile", type=float, default=5.0)
    parser.add_argument("--upper-percentile", type=float, default=95.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--averaged-attention", action="store_true", help="Use the existing head-averaged attention path")
    parser.add_argument("--save-attentions", action="store_true", help="Also save raw per-layer attention tensors")
    args = parser.parse_args()
    if min(args.image_size) <= 0:
        parser.error("image-size must be positive")
    if not all(0 <= ratio <= 1 for ratio in args.ratios):
        parser.error("ratios must be in [0, 1]")
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Require 0 <= lower-percentile < upper-percentile <= 100")
    if not 0 <= args.overlay_alpha <= 1:
        parser.error("overlay-alpha must be in [0, 1]")
    return args


def main():
    args = parse_args()
    # Configure before constructing the backbone or creating CUDA tensors.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["XFORMERS_DISABLED"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model, config = load_model(args.checkpoint, args.backbone, args.device)
    if isinstance(model.backbone, DinoV2) and any(s % model.backbone.patch_size for s in args.image_size):
        raise ValueError("DINOv2 input dimensions must be divisible by its actual patch size")
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Matches VPRDataModule validation: uint8 bicubic resize, then float scaling.
    rgb_transform = T.Compose([
        T.Resize(tuple(args.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])
    for image_index, image_path in enumerate(args.image):
        rgb = rgb_transform(torchvision.io.decode_image(image_path, mode="RGB"))
        assert rgb.min() >= 0 and rgb.max() <= 1
        normalized = T.Normalize(mean, std)(rgb).unsqueeze(0).to(args.device)
        result = extract_attention_map(
            model, normalized, return_head_attn=not args.averaged_attention,
            return_attentions=args.save_attentions,
            lower_percentile=args.lower_percentile, upper_percentile=args.upper_percentile,
        )
        tensors = {key: value.cpu() if isinstance(value, torch.Tensor) else value
                   for key, value in result.items() if key != "attentions"}
        if args.save_attentions:
            tensors["attentions"] = tuple(a.cpu() for a in result["attentions"])
        output = args.output_dir / f"{image_index:03d}_{image_path.stem}"
        output.mkdir(parents=True, exist_ok=True)
        heatmap = colorize(tensors["pixel_attention_map"][0])
        token_display = torch.nn.functional.interpolate(
            tensors["normalized_token_attention_map"][:, None],
            size=tuple(args.image_size), mode="nearest",
        )[0, 0]
        panels = [(rgb, "RGB input"), (colorize(token_display), "Token map (display normalization)"),
                  (heatmap, "Bilinear heatmap (display only)"),
                  ((1 - args.overlay_alpha) * rgb + args.overlay_alpha * heatmap, "Heatmap overlay")]
        mask_stats = []
        tensors["masks"] = {}
        token_h, token_w = result["token_grid"]
        for ratio_index, ratio in enumerate(args.ratios):
            for mode in ["raw_topk", "connected_topk"]:
                token_mask = build_attention_mask(tensors["token_attention_map"], ratio, mode)
                pixel_mask = upsample_token_mask(token_mask, args.image_size)
                key = f"{ratio_index:02d}_{mode}_{ratio:g}"
                tensors["masks"][key] = {"token_mask": token_mask, "pixel_mask": pixel_mask}
                selected = int(token_mask.sum())
                pixel_ratio = pixel_mask.float().mean().item()
                binary = pixel_mask[0].float().expand(3, -1, -1)
                to_pil(binary).save(output / f"{key}_mask.png")
                panels.extend([(binary, f"{mode}, requested {ratio:.1%}\n{selected}/{token_h * token_w} tokens"),
                               (overlay_mask(rgb, pixel_mask[0], args.overlay_alpha),
                                f"{mode} overlay\npixel coverage {pixel_ratio:.2%}")])
                mask_stats.append({"key": key, "mode": mode, "requested_ratio": ratio,
                                   "selected_tokens": selected, "token_ratio": selected / (token_h * token_w),
                                   "pixel_ratio": pixel_ratio})
        torch.save(tensors, output / "attention_maps.pt")
        save_montage(panels, output / "visualization.png", args.image_size)
        metadata = {
            "name": "Attention Proposal Map", "image": str(image_path.resolve()),
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
            "seed": args.seed, "device": args.device, "dtype": "float32",
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "model": config, "image_size": args.image_size, "token_grid": result["token_grid"],
            "aggregation": "mean heads -> mean queries -> mean layers",
            "head_averaged_input": args.averaged_attention,
            "normalization_percentiles": [args.lower_percentile, args.upper_percentile],
            "mask_ranking": "raw token_attention_map (before percentile clipping)",
            "mask_interpolation": "nearest", "display_interpolation": "bilinear",
            "preprocessing": "RGB uint8 -> bicubic resize (antialias) -> float32 [0,1] -> ImageNet Normalize",
            "mean": mean, "std": std, "overlay_alpha": args.overlay_alpha,
            "xformers_enabled": False, "deterministic_algorithms": True,
            "primary_mask_mode": "connected_topk", "masks": mask_stats,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {output}: grid={token_h}x{token_w}, descriptor={tuple(result['descriptor'].shape)}")


if __name__ == "__main__":
    main()
