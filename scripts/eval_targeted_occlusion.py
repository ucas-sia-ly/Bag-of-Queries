"""Frozen BoQ: clean vs shape-matched random vs Attention Proposal Map erasing.

Run 100 queries first, inspect correctness, then 500 using the same reference
cache and seed. The selected query order is a deterministic nested permutation.
No reference truncation, training, or causal map is used.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T
import yaml

from src.analysis import build_attention_mask, extract_attention_map, upsample_token_mask
from src.analysis.perturb import (
    NoLegalTranslation, make_shape_matched_random_mask, make_token_aligned_random_mask,
    mask_iou, perturb_rgb,
)
from src.analysis.retrieval import (
    ATTENTION_CONDITIONS, retrieval_metrics, summarize_ablations, ablation_contrasts,
)
from src.dataloaders import MapillarySLSDataset, PittsburghDataset
from scripts.visualize_attention import colorize, load_model, save_montage

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def reference_manifest(dataset):
    """Ordered names plus file size/mtime detect dataset/cache mismatches."""
    digest = hashlib.sha256()
    for relative in dataset.image_paths[:dataset.num_references]:
        path = dataset.dataset_path / str(relative)
        stat = path.stat()
        digest.update(json.dumps([str(relative), stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


@torch.no_grad()
def get_reference_descriptors(model, dataset, normalize, args, identity):
    path = args.reference_cache
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["identity"] != identity:
            raise ValueError("Reference cache identity mismatch; use a different --reference-cache path")
        descriptors = cached["descriptors"]
        if descriptors.shape != (dataset.num_references, identity["model"]["descriptor_dim"]):
            raise ValueError("Reference cache shape mismatch")
        print(f"Reusing all {len(descriptors)} reference descriptors: {path}", flush=True)
        return descriptors, True
    loader = DataLoader(
        Subset(dataset, range(dataset.num_references)), batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=args.device.startswith("cuda"),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )
    chunks, count = [], 0
    started = time.monotonic()
    for images, indices in loader:
        assert indices.tolist() == list(range(count, count + len(indices)))
        values, _ = model(normalize(images.to(args.device)))
        chunks.append(values.detach().cpu())
        count += len(images)
        if len(chunks) % 20 == 0 or count == dataset.num_references:
            print(f"References {count}/{dataset.num_references}, {time.monotonic() - started:.1f}s", flush=True)
    descriptors = torch.cat(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"identity": identity, "descriptors": descriptors}, temporary)
    temporary.replace(path)
    return descriptors, False


def make_record(query_id, image_path, condition, ratio, repeat, clean, metrics, *, seed,
                area=0., iou=0., paired_eligible=True, alternatives=0, offset=(0, 0),
                placement_seed=0, reason="", mask_pixels=0, mask_mode="none",
                mask_tokens=0, source_mask_tokens=0, token_offset=(None, None),
                sampling="none", unique_placements=0, eligible_random_token=False,
                eligible_random_pixel=False, token_unavailable_reason="", pixel_unavailable_reason=""):
    return {
        "query_id": query_id, "image_path": image_path, "condition": condition,
        "mask_ratio": ratio, "random_repeat": repeat,
        "clean_rank": clean["rank"], "perturbed_rank": metrics["rank"],
        "clean_positive_sim": clean["positive_sim"], "perturbed_positive_sim": metrics["positive_sim"],
        "clean_negative_sim": clean["negative_sim"], "perturbed_negative_sim": metrics["negative_sim"],
        "clean_margin": clean["margin"], "perturbed_margin": metrics["margin"],
        "margin_drop": clean["margin"] - metrics["margin"],
        "positive_similarity_drop": clean["positive_sim"] - metrics["positive_sim"],
        "descriptor_drift": 0.0 if condition == "clean" else metrics["descriptor_drift"],
        "rank_degradation": metrics["rank"] - clean["rank"],
        "hit_at_1": metrics["hit_at_1"], "hit_at_5": metrics["hit_at_5"], "hit_at_10": metrics["hit_at_10"],
        "failure_flip": int(clean["hit_at_1"] and not metrics["hit_at_1"]),
        "mask_area_actual": area, "mask_pixels": mask_pixels, "mask_attention_iou": iou,
        "seed": seed, "placement_seed": placement_seed, "shift_y": offset[0], "shift_x": offset[1],
        "paired_eligible": paired_eligible, "legal_random_locations": alternatives, "exclusion_reason": reason,
        "mask_mode": mask_mode, "mask_tokens": mask_tokens, "source_mask_tokens": source_mask_tokens,
        "token_shift_y": token_offset[0], "token_shift_x": token_offset[1],
        "sampling": sampling, "unique_placements": unique_placements,
        "eligible_random_token": eligible_random_token, "eligible_random_pixel": eligible_random_pixel,
        "token_unavailable_reason": token_unavailable_reason, "pixel_unavailable_reason": pixel_unavailable_reason,
    }


def build_ablation_variants(rgb, token_map, query_id, args):
    """Build each construction and its own random controls from the same scores.

    Require uniform nearest cells for strict pixel area/shape equality. DINOv2
    patch-divisible resolutions satisfy this. Unsupported resolutions fail
    explicitly instead of silently accepting a pixel-area confound.
    """
    token_h, token_w = token_map.shape[-2:]
    height, width = rgb.shape[-2:]
    if height % token_h or width % token_w:
        raise ValueError("Strict ablations require RGB dimensions divisible by the actual token grid")
    scale_y, scale_x = height // token_h, width // token_w
    variants, specs = [], []
    fill_rgb = MEAN if args.fill_source == "imagenet_mean" else None
    for mode in args.mask_modes:
        for ratio in args.ratios:
            token_mask = build_attention_mask(token_map.cpu(), ratio, mode)[0]
            budget = round(ratio * token_h * token_w)
            assert int(token_mask.sum()) == budget
            stream = [args.seed, query_id, int(round(ratio * 1e6))]
            if mode != "connected_topk":
                stream.append(71)
            seeds = {"random_pixel": int(np.random.SeedSequence(stream).generate_state(1)[0]),
                     "random_token": int(np.random.SeedSequence(stream + [97]).generate_state(1)[0])}
            groups, reasons = {}, {"random_pixel": "not requested", "random_token": "not requested"}
            # Token translations are selected BEFORE attention-mask upsampling.
            if "random_token" in args.random_baselines:
                try:
                    tokens, offsets, count = make_token_aligned_random_mask(
                        token_mask, args.random_repeats, seed=seeds["random_token"])
                    assert (tokens.sum((1, 2)) == budget).all()
                    pixels = upsample_token_mask(tokens, (height, width)).to(rgb.device)
                    pixel_offsets = [(dy * scale_y, dx * scale_x) for dy, dx in offsets]
                    sampling = "without_replacement" if count >= args.random_repeats else "all_once_then_replacement"
                    groups["random_token"] = (pixels, pixel_offsets, offsets, count, sampling)
                    reasons["random_token"] = ""
                except NoLegalTranslation as error:
                    reasons["random_token"] = str(error)
            attention_mask = upsample_token_mask(token_mask, (height, width)).to(rgb.device)
            assert int(attention_mask.sum()) == budget * scale_y * scale_x
            if "random_pixel" in args.random_baselines:
                try:
                    pixels, offsets, count = make_shape_matched_random_mask(
                        attention_mask, args.random_repeats, seed=seeds["random_pixel"])
                    groups["random_pixel"] = (pixels, offsets, [(None, None)] * args.random_repeats, count, "with_replacement")
                    reasons["random_pixel"] = ""
                except NoLegalTranslation as error:
                    reasons["random_pixel"] = str(error)
            common = dict(mask_mode=mode, source_mask_tokens=budget,
                          eligible_random_token="random_token" in groups,
                          eligible_random_pixel="random_pixel" in groups,
                          token_unavailable_reason=reasons["random_token"],
                          pixel_unavailable_reason=reasons["random_pixel"])
            pending = [(ATTENTION_CONDITIONS[mode], -1, attention_mask, (0, 0), (0, 0),
                        0, "none", 0, 0, args.primary_baseline in groups)]
            for baseline in args.random_baselines:
                if baseline not in groups:
                    continue
                pixels, offsets, token_offsets, count, sampling = groups[baseline]
                assert (pixels.sum((1, 2)) == attention_mask.sum()).all()
                for repeat, (mask, offset, token_offset) in enumerate(zip(pixels, offsets, token_offsets)):
                    pending.append((baseline, repeat, mask, offset, token_offset, count,
                                    sampling, len(set(offsets)), seeds[baseline], True))
            for condition, repeat, mask, offset, token_offset, count, sampling, unique, seed, eligible in pending:
                variants.append(perturb_rgb(rgb, mask, operator=args.operator, fill_rgb=fill_rgb,
                                            blur_kernel=args.blur_kernel, blur_sigma=args.blur_sigma)[0])
                specs.append(dict(common, condition=condition, ratio=ratio, repeat=repeat,
                                  area=mask.float().mean().item(), iou=mask_iou(mask, attention_mask),
                                  mask_pixels=int(mask.sum()), mask_tokens=None if condition == "random_pixel" else budget,
                                  offset=offset, token_offset=token_offset, alternatives=count, sampling=sampling,
                                  unique_placements=unique, placement_seed=seed, paired_eligible=eligible,
                                  reason="" if eligible else reasons[args.primary_baseline]))
    return variants, specs


def plot_results(output, summaries, pairs, primary_ratio, *, target_label="attention", random_label="random"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "visualizations"
    folder.mkdir(exist_ok=True)
    primary = [p for p in pairs if p["mask_ratio"] == primary_ratio]
    for metric, label in [("margin_drop", "Margin drop (primary endpoint)"),
                          ("descriptor_drift", "Cosine descriptor drift")]:
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        if primary:
            x = np.array([p[f"random_{metric}"] for p in primary])
            y = np.array([p[f"attention_{metric}"] for p in primary])
            ax.scatter(x, y, s=18, alpha=.55)
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(xlabel=f"{random_label} (within-query mean)", ylabel=target_label,
               title=f"{label} | ratio={primary_ratio:.0%}, paired n={len(primary)}")
        fig.savefig(folder / f"{metric}_attention_vs_random.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for condition in ["attention", "random"]:
        values = np.sort([p[f"{condition}_rank_degradation"] for p in primary])
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    label=target_label if condition == "attention" else random_label)
    ax.set(xlabel="Best-positive rank degradation (random: within-query mean)", ylabel="ECDF",
           title=f"Rank degradation | ratio={primary_ratio:.0%}")
    ax.set_xscale("symlog", linthresh=1)
    if primary:
        ax.legend()
    fig.savefig(folder / "rank_degradation_ecdf.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, k in zip(axes, [1, 5, 10]):
        for condition in ["clean", "random", "attention"]:
            subset = sorted([s for s in summaries if s["condition"] == condition], key=lambda s: s["mask_ratio"])
            label = {"attention": target_label, "random": random_label, "clean": "clean"}[condition]
            ax.plot([100 * s["mask_ratio"] for s in subset], [s[f"hit_at_{k}"] for s in subset], "o-", label=label)
        ax.set(xlabel="Mask ratio (%)", ylabel=f"Recall@{k}", ylim=(0, 1))
    axes[0].legend()
    fig.suptitle("Recall on matched queries per ratio (random: per-query repeat mean)")
    fig.savefig(folder / "recall_at_k_vs_mask_ratio.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for condition in ["attention", "random"]:
        subset = sorted([s for s in summaries if s["condition"] == condition], key=lambda s: s["mask_ratio"])
        ax.plot([100 * s["mask_ratio"] for s in subset], [s["failure_flip"] for s in subset], "o-",
                label=target_label if condition == "attention" else random_label)
    ax.set(xlabel="Mask ratio (%)", ylabel="Clean-correct to incorrect / all matched queries",
           title="Failure flip rate (random: within-query repeat mean)", ylim=(0, 1))
    ax.legend()
    fig.savefig(folder / "failure_flip_rate.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", choices=["msls-val", "pitts30k-val"], required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--image-size", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--ratios", type=float, nargs="+", default=[.10, .15, .20])
    parser.add_argument("--primary-ratio", type=float, default=.15)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--mask-modes", nargs="+", choices=list(ATTENTION_CONDITIONS),
                        default=["connected_topk", "raw_topk"])
    parser.add_argument("--random-baselines", nargs="+", choices=["random_pixel", "random_token"],
                        default=["random_pixel", "random_token"])
    parser.add_argument("--primary-baseline", choices=["random_pixel", "random_token"], default="random_token")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--operator", choices=["mean_fill", "gaussian_blur"], default="mean_fill")
    parser.add_argument("--fill-source", choices=["image_mean", "imagenet_mean"], default="image_mean")
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--blur-sigma", type=float, default=5.)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--cases-per-category", type=int, default=3)
    parser.add_argument("--case-threshold", type=float, default=.01, help="Absolute paired margin-drop difference for case categories")
    args = parser.parse_args()
    if (args.num_queries < 1 or args.random_repeats < 1 or args.batch_size < 1 or args.workers < 0
            or min(args.image_size) < 1 or args.cases_per_category < 0 or args.bootstrap_samples < 0):
        parser.error("Invalid counts, batch size, workers, or resolution")
    if (len(set(args.ratios)) != len(args.ratios) or not all(0 < r < 1 for r in args.ratios)
            or args.primary_ratio not in args.ratios):
        parser.error("ratios must be unique in (0,1) and include primary-ratio")
    if not np.isfinite(args.case_threshold) or args.case_threshold < 0:
        parser.error("case-threshold must be finite and nonnegative")
    if (args.primary_baseline not in args.random_baselines or len(set(args.mask_modes)) != len(args.mask_modes)
            or len(set(args.random_baselines)) != len(args.random_baselines)):
        parser.error("Require unique modes/baselines and inclusion of primary-baseline")
    args.reference_cache = args.reference_cache or args.output_dir / "reference_descriptors.pt"
    return args


def plot_ablation_results(output, summaries, pairs, contrasts, args):
    """Separate baseline cohorts, and explicit common-query ablation contrasts."""
    for mode in args.mask_modes:
        for baseline in args.random_baselines:
            folder = output / "comparisons" / mode / baseline
            folder.mkdir(parents=True, exist_ok=True)
            names = {"clean": "clean", baseline: "random", ATTENTION_CONDITIONS[mode]: "attention"}
            group = [dict(s, condition=names[s["condition"]]) for s in summaries
                     if s["mask_mode"] == mode and s["baseline"] == baseline]
            paired = [p for p in pairs if p["mask_mode"] == mode and p["baseline"] == baseline]
            plot_results(folder, group, paired, args.primary_ratio,
                         target_label=ATTENTION_CONDITIONS[mode], random_label=baseline)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "visualizations"
    folder.mkdir(exist_ok=True)
    construction = [c for c in contrasts if c["comparison"] == "attention_raw_topk-minus-attention_connected"]
    if construction:
        construction.sort(key=lambda c: c["mask_ratio"])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, metric, label in zip(axes, ["margin_drop", "hit_at_1", "failure_flip"],
                                    ["Margin drop", "Recall@1", "Failure flip rate"]):
            for side, condition in [("left", "attention_raw_topk"), ("right", "attention_connected")]:
                ax.plot([100 * c["mask_ratio"] for c in construction],
                        [c[f"{side}_{metric}"] for c in construction], "o-", label=condition)
            ax.set(xlabel="Mask ratio (%)", ylabel=label)
        axes[0].legend()
        fig.suptitle("Mask construction: identical queries, token budgets and pixel areas")
        fig.savefig(folder / "mask_construction.png", dpi=160)
        plt.close(fig)
    for mode in args.mask_modes:
        primary = {b: {p["query_id"]: p for p in pairs if p["mask_mode"] == mode
                       and p["baseline"] == b and p["mask_ratio"] == args.primary_ratio}
                   for b in ["random_pixel", "random_token"]}
        ids = sorted(primary["random_pixel"].keys() & primary["random_token"].keys())
        if not ids:
            continue
        x = np.array([primary["random_pixel"][q]["random_margin_drop"] for q in ids])
        y = np.array([primary["random_token"][q]["random_margin_drop"] for q in ids])
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.scatter(x, y, alpha=.5, s=18)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlabel="random_pixel mean margin drop", ylabel="random_token mean margin drop",
               title=f"{mode}: common queries n={len(ids)}, ratio={args.primary_ratio:.0%}")
        fig.savefig(folder / f"{mode}_random_alignment.png", dpi=160)
        plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "per_query.csv").exists():
        raise FileExistsError("Output already contains per_query.csv; choose a new --output-dir")
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rgb_transform = T.Compose([
        T.Resize(tuple(args.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])
    dataset_class = MapillarySLSDataset if args.dataset == "msls-val" else PittsburghDataset
    dataset = dataset_class(args.dataset_path, transform=rgb_transform)
    normalize = T.Normalize(MEAN, STD)
    model, model_config = load_model(args.checkpoint, args.backbone, args.device)
    root = Path(__file__).resolve().parents[1]
    identity = {
        "checkpoint_sha256": file_hash(args.checkpoint), "dataset_path": str(args.dataset_path.resolve()),
        "reference_manifest_sha256": reference_manifest(dataset), "num_references": dataset.num_references,
        "image_size": args.image_size, "mean": MEAN, "std": STD,
        "preprocessing": "RGB uint8 bicubic antialias -> float32 scale -> ImageNet Normalize",
        "model": model_config, "dtype": "float32", "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "device": args.device,
        "batch_size": args.batch_size, "tf32": False, "xformers": False,
        "model_source_sha256": {p: file_hash(root / p) for p in ["src/boq.py", "src/backbones.py", "hubconf.py"]},
    }
    valid_queries = [q for q in range(dataset.num_queries) if 0 < len(np.unique(dataset.ground_truth[q])) < dataset.num_references]
    if args.num_queries > len(valid_queries):
        raise ValueError(f"Requested {args.num_queries} queries but only {len(valid_queries)} have valid ground truth")
    query_ids = np.random.default_rng(args.seed).permutation(valid_queries)[:args.num_queries].tolist()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(reference_identity=identity, query_ids=query_ids,
                  query_sampling="seeded permutation prefix; nested subsets", total_queries=dataset.num_queries,
                  invalid_ground_truth_queries=dataset.num_queries - len(valid_queries),
                  aggregation="mean heads -> mean queries -> mean layers", attention_name="Attention Proposal Map",
                  schema_version=2, mask="connected_topk and/or raw_topk; exact round(ratio*Ht*Wt)",
                  random_pixel="uniform integer-pixel translations, with replacement, excluding original position",
                  random_token="translate original token mask; without replacement if enough positions; otherwise all once then replacement; nearest upsample",
                  translation_unavailable="retain target for direct construction comparison; exclude only from affected shape/baseline paired cohort",
                  pixel_area_policy="require RGB dimensions divisible by actual token grid; assert exact area",
                  primary_endpoint="margin_drop", random_analysis_unit="within-query mean, then paired queries",
                  rank_ties="ascending reference index", bootstrap_unit="query",
                  mask_area_actual_units="fraction of resized RGB pixels", precision="float32",
                  fill_definition="original clean image channel mean" if args.fill_source == "image_mean" else MEAN)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    references, cache_reused = get_reference_descriptors(model, dataset, normalize, args, identity)
    if not torch.isfinite(references).all():
        raise ValueError("Non-finite reference descriptors")
    torch.testing.assert_close(references.norm(dim=1), torch.ones(len(references)), atol=1e-5, rtol=1e-5)
    references = references.to(args.device)
    rows, cases, seen = [], {}, {}
    case_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 871]))
    start = time.monotonic()
    for query_number, query_id in enumerate(query_ids):
        rgb, _ = dataset[dataset.num_references + query_id]
        rgb = rgb.to(args.device)
        result = extract_attention_map(model, normalize(rgb).unsqueeze(0), return_head_attn=False)
        clean_descriptor = result["descriptor"]
        variants, specs = build_ablation_variants(rgb, result["token_attention_map"], query_id, args)
        descriptors = [clean_descriptor]
        for first in range(0, len(variants), args.batch_size):
            values, _ = model(normalize(torch.stack(variants[first:first + args.batch_size])))
            descriptors.append(values)
        metrics = retrieval_metrics(torch.cat(descriptors), references, dataset.ground_truth[query_id], clean_descriptor)
        clean = metrics[0]
        image_path = str(dataset.image_paths[dataset.num_references + query_id])
        rows.append(make_record(query_id, image_path, "clean", 0., -1, clean, clean, seed=args.seed))
        query_rows = []
        for spec, values in zip(specs, metrics[1:]):
            query_rows.append(make_record(query_id, image_path, clean=clean, metrics=values, seed=args.seed, **spec))
        rows.extend(query_rows)
        # Reservoir sample cases by the paired PRIMARY margin endpoint, not by extreme values.
        case_mode = "connected_topk" if "connected_topk" in args.mask_modes else args.mask_modes[0]
        target_indices = [i for i, s in enumerate(specs) if s["ratio"] == args.primary_ratio
                          and s["condition"] == ATTENTION_CONDITIONS[case_mode]]
        target_index = target_indices[0]
        target = query_rows[target_index]
        draw_indices = [i for i, r in enumerate(query_rows) if r["mask_ratio"] == args.primary_ratio
                        and r["condition"] == args.primary_baseline and r["mask_mode"] == case_mode]
        draws = [query_rows[i] for i in draw_indices]
        if draws and args.cases_per_category:
            random_damage = float(np.mean([r["margin_drop"] for r in draws]))
            difference = target["margin_drop"] - random_damage
            category = "attention_stronger" if difference > args.case_threshold else (
                "random_stronger" if difference < -args.case_threshold else "almost_identical")
            seen[category] = seen.get(category, 0) + 1
            cases.setdefault(category, [])
            slot = len(cases[category]) if len(cases[category]) < args.cases_per_category else int(case_rng.integers(seen[category]))
            if slot < args.cases_per_category:
                data = {
                    "query_id": query_id, "image_path": image_path, "category": category,
                    "paired_margin_difference": difference, "target": target, "random_draws": draws,
                    "random_mean_margin_drop": random_damage,
                }
                panels = [(rgb.cpu(), f"Clean | best positive rank {clean['rank']}"),
                          (colorize(result["pixel_attention_map"][0].cpu()), "Attention Proposal Map"),
                          (variants[target_index].cpu(), f"Target | rank {target['perturbed_rank']}\nmargin drop {target['margin_drop']:.4f}"),
                          (variants[draw_indices[0]].cpu(), f"{args.primary_baseline} #0 | rank {draws[0]['perturbed_rank']}\nmean of draws: {random_damage:.4f}")]
                entry = (data, panels)
                if slot == len(cases[category]):
                    cases[category].append(entry)
                else:
                    cases[category][slot] = entry
        if (query_number + 1) % 10 == 0 or query_number + 1 == len(query_ids):
            print(f"Queries {query_number + 1}/{len(query_ids)}, {time.monotonic() - start:.1f}s", flush=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    summaries, pairs = summarize_ablations(rows, args.ratios, args.mask_modes, args.random_baselines,
                                          repeats=args.random_repeats, seed=args.seed,
                                          bootstrap_samples=args.bootstrap_samples)
    write_csv(args.output_dir / "summary.csv", summaries)
    if pairs:
        write_csv(args.output_dir / "paired_query.csv", pairs)
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    clean_summary = {"n_queries": len(clean_rows), "num_references": dataset.num_references,
                     "reference_cache_reused": cache_reused,
                     **{f"Recall@{k}": float(np.mean([r[f"hit_at_{k}"] for r in clean_rows])) for k in [1, 5, 10]}}
    (args.output_dir / "clean_metrics.json").write_text(json.dumps(clean_summary, indent=2) + "\n")
    contrast_summary, contrast_pairs = ablation_contrasts(rows, pairs, args.ratios, seed=args.seed,
                                                        bootstrap_samples=args.bootstrap_samples)
    if contrast_summary:
        write_csv(args.output_dir / "ablation_summary.csv", contrast_summary)
        write_csv(args.output_dir / "ablation_paired_query.csv", contrast_pairs)
    plot_ablation_results(args.output_dir, summaries, pairs, contrast_summary, args)
    case_folder = args.output_dir / "visualizations" / "cases"
    case_folder.mkdir(exist_ok=True)
    for category, entries in cases.items():
        for data, panels in entries:
            prefix = case_folder / f"{category}_query{data['query_id']}"
            save_montage(panels, prefix.with_suffix(".png"), args.image_size)
            prefix.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    (case_folder / "categories.json").write_text(json.dumps({
        "definition": f"target margin_drop - per-query mean {args.primary_baseline} margin_drop",
        "threshold": args.case_threshold, "sampling": "seeded reservoir within category",
        "counts": {c: seen.get(c, 0) for c in ["attention_stronger", "random_stronger", "almost_identical"]},
        "saved": {c: len(cases.get(c, [])) for c in ["attention_stronger", "random_stronger", "almost_identical"]},
    }, indent=2) + "\n")
    print(json.dumps(clean_summary), flush=True)
    for row in summaries:
        if row["mask_ratio"] == args.primary_ratio and row["baseline"] == args.primary_baseline:
            print(json.dumps(row), flush=True)
    print(f"Completed {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
