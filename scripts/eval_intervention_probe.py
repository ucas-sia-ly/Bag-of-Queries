"""Audit local attention faithfulness using batched token-window deletions.

The source experiment config fixes checkpoint, preprocessing, reference identity
and query order. References must already be cached; this script never recomputes
them. Example: --source-config outputs/stage1/ablation500/config.yaml
--output-dir outputs/stage1/intervention100
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.transforms import v2 as T
import yaml

from scripts.eval_targeted_occlusion import MEAN, STD, file_hash, reference_manifest
from scripts.visualize_attention import load_model
from src.analysis.intervention import (
    probe_query, project_window_values, summarize_faithfulness, summarize_query,
)
from src.dataloaders import MapillarySLSDataset


CATEGORIES = ("attention_strong_intervention_strong", "attention_strong_intervention_weak",
              "attention_weak_intervention_strong")


def collect_cases(rows, cases, counts, rng, capacity):
    """Uniform query reservoir; random qualifying window within each query.

Strong/weak means upper/lower within-query quintile. Strong intervention also
requires positive signed damage. Cases illustrate disagreement, not prevalence.
"""
    a = np.array([r["attention_score"] for r in rows])
    d = np.array([r["margin_drop"] for r in rows])
    ah, al = a >= np.quantile(a, .8), a <= np.quantile(a, .2)
    dh, dl = (d >= np.quantile(d, .8)) & (d > 0), d <= np.quantile(d, .2)
    for category, qualifies in zip(CATEGORIES, [ah & dh, ah & dl, al & dh]):
        indices = np.flatnonzero(qualifies)
        counts[category]["qualifying_windows"] += len(indices)
        if not len(indices):
            continue
        counts[category]["eligible_queries"] += 1
        chosen = int(rng.choice(indices))
        entry = dict(rows[chosen], category=category, window_index=chosen)
        bucket = cases[category]
        slot = len(bucket) if len(bucket) < capacity else int(rng.integers(counts[category]["eligible_queries"]))
        if slot < capacity:
            if slot == len(bucket):
                bucket.append(entry)
            else:
                bucket[slot] = entry


def plot_query(rgb, maps, query, path, window_size, selected=None):
    """Raw attention and signed deletion maps; overlap projection is display only."""
    rgb = rgb.permute(1, 2, 0).numpy()
    raw = maps["token_attention"]
    positions, damage = maps["positions"], maps["margin_drop"]
    projected = maps["deletion_display"]
    h, w = raw.shape
    height, width = rgb.shape[:2]
    extent = [0, width, height, 0]
    scale = max(float(np.max(np.abs(damage))), 1e-8)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB input / selected 2D window")
    im = axes[0, 1].imshow(raw, extent=extent, cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title("Attention Proposal Map (raw aggregated)")
    fig.colorbar(im, ax=axes[0, 1], shrink=.8)
    # Values are plotted at window centers, not assigned to individual tokens.
    im = axes[0, 2].scatter((positions[:, 1]+window_size/2)*width/w,
                            (positions[:, 0]+window_size/2)*height/h,
                            c=damage, s=18, marker="s", cmap="coolwarm", norm=norm)
    axes[0, 2].set(xlim=(0, width), ylim=(height, 0), title="Signed deletion sensitivity (window centers)")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(im, ax=axes[0, 2], shrink=.8, label="Clean margin − deleted margin")
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(raw, extent=extent, cmap="viridis", alpha=.5, interpolation="nearest")
    axes[1, 0].set_title("Attention overlay")
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(projected, extent=extent, cmap="coolwarm", norm=norm, alpha=.55, interpolation="nearest")
    axes[1, 1].set_title("Deletion overlay (mean of covering windows)")
    ax = axes[1, 2]
    ax.scatter(maps["attention_score"], damage, s=9, alpha=.5)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set(xlabel="Raw mean attention in window", ylabel="Signed margin drop", title="Within-query attention vs damage")
    if selected is not None:
        left, top = selected["window_x"]*width/w, selected["window_y"]*height/h
        for axis in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
            axis.add_patch(Rectangle((left, top), window_size*width/w, window_size*height/h,
                                    fill=False, edgecolor="#00ff00", linewidth=2))
        ax.scatter([selected["attention_score"]], [selected["margin_drop"]], marker="*", s=180,
                   edgecolor="black", color="#00ff00", zorder=5)
    for axis in [*axes[0], axes[1, 0], axes[1, 1]]:
        axis.set_xticks([]); axis.set_yticks([])
    rho = query["spearman"]
    title = f"Query {query['query_id']} | rho={rho:.3f}" if rho is not None else f"Query {query['query_id']} | rho undefined"
    if selected is not None:
        title += f"\n{selected['category']} | selected damage={selected['margin_drop']:.5f}"
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(output, summary, queries, config, counts):
    rho, diff = summary["spearman"], summary["top_minus_bottom_damage"]
    percentile = summary["attention_max_damage_percentile"]
    def interval(s, key="mean"):
        return "undefined" if s is None else f"{s[key]:.6f} [{s[key+'_ci_low']:.6f}, {s[key+'_ci_high']:.6f}]"
    text = ["# Attention Faithfulness：Intervention Audit", "", f"**{summary['conclusion']}**", "",
            "## 固定 protocol", "",
            f"同一 checkpoint、MSLS 验证集和保存的 query 顺序前 {len(queries)} 个；"
            f"完整 reference database 共 {config['reference_identity']['num_references']} 张，仅复用缓存，没有重算 reference。",
            f"真实 backbone token grid 动态生成 {config['window_size']}×{config['window_size']} windows，stride={config['stride']}。"
            "token mask → nearest upsample → RGB [0,1] mean-fill → ImageNet Normalize → frozen model → full-database retrieval。",
            f"Mean fill={config['fill_source']}；variant batch_size={config['batch_size']}，"
            f"共 {sum(q['n_windows'] for q in queries)} windows、{sum(q['variant_forward_calls'] for q in queries)} 次 batched variant forward。",
            "Attention score 是每个 window 内 raw aggregated attention 的均值；聚合仍为 mean heads → mean queries → mean layers，未增加权重。"
            "margin_drop = clean margin − deleted margin，保留负数。Attention Proposal Map 没有改名为 causal map。", "",
            "## Query-level faithfulness", "",
            f"独立单位是 query。每个 query 单独计算 Spearman，未将 windows 汇总为独立样本，也未报告 window-level p-value。"
            f"有效相关 query={summary['n_spearman_defined']}，常量导致未定义={summary['n_spearman_undefined']}。"
            "相关统计及 positive fraction 以有效相关 query 为分母；top/bottom 和 percentile 统计使用全部 query。",
            f"固定 seed={config['seed']}，{config['bootstrap_samples']:,} 次 query-level percentile bootstrap；"
            "表中区间均为逐项 95% CI，不是 simultaneous CI。重采样同时保留同一 query 的 top/bottom 配对。", "",
            "| Endpoint | Estimate [95% CI] |", "| --- | --- |",
            f"| Mean Spearman | {interval(rho)} |",
            f"| Median Spearman | {interval(rho, 'median')} |",
            f"| Positive-correlation query fraction | {interval(rho, 'positive_fraction')} |",
            f"| Mean top − bottom attention-window damage | {interval(diff)} |",
            f"| Median top − bottom damage | {interval(diff, 'median')} |",
            f"| Mean attention-max damage percentile | {interval(percentile)} |",
            f"| Median attention-max damage percentile | {interval(percentile, 'median')} |", "",
            f"Across-query mean top damage={np.mean([q['top_mean_damage'] for q in queries]):.6f}；"
            f"mean bottom damage={np.mean([q['bottom_mean_damage'] for q in queries]):.6f}。",
            f"Top/bottom 分别取 K=max(1, round({config['tail_fraction']}×window_count)) 个 windows。"
            "按 raw attention 排序，ties 按 row-major 坐标；同一 query 先求两组 mean damage 的差再 bootstrap。"
            "attention-max 遇到 ties 选 row-major 第一个，同时保存 ties 数量。"
            "Deletion-damage percentile = 100×(ascending average rank − 1)/(N − 1)，100 表示 damage 最高，0 最低。", "",
            "## 解释边界", "",
            f"描述性判断规则：mean Spearman ≥ {config['min_mean_spearman']}、"
            "mean Spearman CI 下界 > 0、top−bottom damage CI 下界 > 0，且相关均可定义时，"
            "称为 useful proposal signal。该可配置强度阈值用于描述结果，不是通用标准或预注册阈值。"
            "即使通过，也不能据此声称 attention 是精确或校准过的局部 sensitivity estimator。",
            "该结果只针对当前模型、mean-fill、窗口尺度和样本。重叠 windows 相互依赖，不能用于扩充统计样本数；"
            "相邻 query 的路线相关性未作 cluster 校正。本步骤不训练、不 fusion，不扩展到后续阶段。", "",
            "## 可视化与案例", "",
            "所有 query 的 raw attention、window attention、signed window damage、坐标和 overlap counts 保存于 maps/*.npz。"
            "overview / cases 图包含 attention map、以 0 为中心的 signed deletion sensitivity、两种 overlay 和 scatter。"
            "Deletion overlay 是覆盖该 token 的 windows 的平均 damage，仅供显示，不是单 token effect，也不用于统计。",
            "三类案例按 query 内 attention / damage 上下 20% 划分；strong intervention 额外要求 damage > 0。"
            "固定 seed，先在每个合格 query 内随机取一个合格 window，再对 query 做 reservoir sampling。"
            "案例只用于人工检查，不能替代全体 query-level 结果；绿色框和散点星号标记被选 window。", "",
            "| Category | Eligible queries | Qualifying windows | Saved cases |", "| --- | --- | --- | --- |"]
    for category, count in counts.items():
        text.append(f"| {category} | {count['eligible_queries']} | {count['qualifying_windows']} | {count['saved']} |")
    if "analysis_revision" in config:
        text += ["", "解释修订：初版阈值 0.10 会把稳定但弱的相关直接归为 useful proposal。"
                 "观察完整结果后，按用户要求区分正向 proposal 信息与局部 faithfulness，描述性阈值改为 0.30。"
                 "这是透明的事后解释选择，不能作为预注册检验；原始阈值与源码哈希保存在 config.yaml。"
                 "两种阈值下 Spearman、top−bottom、CI 和 percentile 完全相同，仅结论措辞改变；没有重跑模型。"
                 "positive_proposal_signal_supported 仍为 true，表示稳定正向关联，不代表局部估计精度足够。"]
    text += ["", "![query-level faithfulness](plots/query_faithfulness.png)", "", "## 复现与验证", "",
             "模型 / checkpoint / reference manifest 与原实验 identity 严格核对，cache SHA-256 在前后验证未改变。"
             "每个 clean query 的 rank 与 margin 与原实验 CSV 对照，结果写入 per_query.csv。"
             "完整运行配置、输入哈希和精度设置见 config.yaml。", "", "```bash",
             f"python scripts/eval_intervention_probe.py --source-config {config['source_config']} "
             f"--output-dir {output} --num-queries {len(queries)} --batch-size {config['batch_size']} "
             f"--bootstrap-samples {config['bootstrap_samples']} --window-size {config['window_size']} "
             f"--stride {config['stride']} --tail-fraction {config['tail_fraction']} "
             f"--min-mean-spearman {config['min_mean_spearman']} --cases-per-category {config['cases_per_category']}", "```", ""]
    (output / "INTERVENTION_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window-size", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tail-fraction", type=float, default=.10)
    p.add_argument("--bootstrap-samples", type=int, default=20000)
    p.add_argument("--min-mean-spearman", type=float, default=.30)
    p.add_argument("--cases-per-category", type=int, default=3)
    a = p.parse_args()
    if (a.num_queries < 1 or a.batch_size < 2 or a.window_size < 1 or a.stride < 1
            or not 0 < a.tail_fraction <= .5 or a.bootstrap_samples < 10000
            or not 0 <= a.min_mean_spearman <= 1 or a.cases_per_category < 1):
        p.error("Invalid probe / bootstrap parameters")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Use an empty output directory to avoid mixing incomplete/different runs")
    return a


@torch.no_grad()
def main():
    args = parse_args()
    source = yaml.safe_load(args.source_config.read_text())
    if source["dataset"] != "msls-val" or source["operator"] != "mean_fill":
        raise ValueError("Primary probe requires the MSLS mean-fill experiment config")
    if source["fill_source"] not in {"image_mean", "imagenet_mean"}:
        raise ValueError("Unknown source mean-fill definition")
    seed, device = source["seed"], source["device"]
    query_ids = source["query_ids"][:args.num_queries]
    if len(query_ids) != args.num_queries or len(set(query_ids)) != len(query_ids):
        raise ValueError("Requested query prefix unavailable or duplicated")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["XFORMERS_DISABLED"] = "1"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Source config paths are repository-root relative, matching its producer.
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    checkpoint, cache = resolve(source["checkpoint"]), resolve(source["reference_cache"])
    dataset_path = resolve(source["dataset_path"])
    if not cache.is_file():
        raise FileNotFoundError("Reference cache required; this audit never recomputes references")
    rgb_transform = T.Compose([T.Resize(tuple(source["image_size"]), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                               T.ToDtype(torch.float32, scale=True)])
    dataset = MapillarySLSDataset(dataset_path, transform=rgb_transform)
    normalize = T.Normalize(MEAN, STD)
    cache_hash = file_hash(cache)
    cached = torch.load(cache, map_location="cpu", weights_only=True)
    identity = source["reference_identity"]
    if cached["identity"] != identity:
        raise ValueError("Reference cache identity differs from source experiment")
    checks = dict(checkpoint_sha256=file_hash(checkpoint), dataset_path=str(dataset_path.resolve()),
                  reference_manifest_sha256=reference_manifest(dataset), num_references=dataset.num_references,
                  image_size=source["image_size"], mean=MEAN, std=STD, dtype="float32", device=device,
                  preprocessing="RGB uint8 bicubic antialias -> float32 scale -> ImageNet Normalize",
                  torch=str(torch.__version__), torchvision=str(torchvision.__version__), tf32=False, xformers=False,
                  model_source_sha256={p: file_hash(root / p) for p in identity["model_source_sha256"]})
    for key, value in checks.items():
        if identity[key] != value:
            raise ValueError(f"Source experiment identity mismatch: {key}")
    model, model_config = load_model(checkpoint, source["backbone"], device)
    if identity["model"] != model_config:
        raise ValueError("Checkpoint/model dimensions differ from source experiment")
    references = cached["descriptors"]
    if (references.dtype != torch.float32 or references.shape != (dataset.num_references, model_config["descriptor_dim"])
            or not torch.isfinite(references).all()):
        raise ValueError("Incomplete or invalid reference descriptors")
    torch.testing.assert_close(references.norm(dim=1), torch.ones(len(references)), atol=1e-5, rtol=1e-5)
    references = references.to(device)
    del cached
    source_csv = args.source_config.parent / "per_query.csv"
    original = pd.read_csv(source_csv, usecols=["query_id", "condition", "clean_rank", "clean_margin"])
    original = original[original.condition == "clean"].set_index("query_id")
    if not original.index.is_unique or original.index.tolist()[:len(query_ids)] != query_ids:
        raise ValueError("Source query order differs between config and CSV")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(seed=seed, device=device, checkpoint=str(checkpoint), reference_cache=str(cache),
                  reference_cache_sha256=cache_hash, reference_identity=identity,
                  source_config_sha256=file_hash(args.source_config), source_csv_sha256=file_hash(source_csv),
                  query_ids=query_ids, image_size=source["image_size"], fill_source=source["fill_source"],
                  source_sha256={p: file_hash(root / p) for p in ["src/analysis/intervention.py", "scripts/eval_intervention_probe.py", "src/analysis/attention.py", "src/analysis/perturb.py", "src/analysis/retrieval.py"]},
                  aggregation="mean heads -> mean queries -> mean layers", bootstrap_unit="query",
                  reference_cache_reused=True, dtype="float32", tf32=False, xformers=False)
    out = args.output_dir
    for folder in [out, out / "maps", out / "plots", out / "visualizations", out / "visualizations/cases"]:
        folder.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cases = {c: [] for c in CATEGORIES}
    counts = {c: dict(eligible_queries=0, qualifying_windows=0) for c in CATEGORIES}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 918]))
    queries, started = [], time.monotonic()
    print(f"Reusing {len(references)} cached references; probing {len(query_ids)} queries in saved order.", flush=True)
    with (out / "per_window.csv").open("w", newline="") as stream:
        writer = None
        for number, query_id in enumerate(query_ids):
            rgb, _ = dataset[dataset.num_references + query_id]
            result = probe_query(model, rgb.to(device), normalize, references, dataset.ground_truth[query_id], query_id,
                                 batch_size=args.batch_size, window_size=args.window_size, stride=args.stride,
                                 fill_rgb=MEAN if source["fill_source"] == "imagenet_mean" else None)
            rows = result["rows"]
            clean_delta = result["clean"]["margin"] - original.loc[query_id, "clean_margin"]
            if result["clean"]["rank"] != original.loc[query_id, "clean_rank"] or abs(clean_delta) > 1e-6:
                raise ValueError(f"Clean retrieval differs from source query {query_id}: delta={clean_delta}")
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
            writer.writerows(rows); stream.flush()
            query = summarize_query(rows, args.tail_fraction)
            query.update(token_h=result["token_grid"][0], token_w=result["token_grid"][1],
                         variant_forward_calls=result["variant_forward_calls"],
                         max_variant_batch=max(result["variant_batch_sizes"]), min_variant_batch=min(result["variant_batch_sizes"]),
                         clean_margin_delta_vs_source=float(clean_delta))
            queries.append(query)
            pd.DataFrame(queries).to_csv(out / "per_query.csv", index=False)
            damage = np.array([r["margin_drop"] for r in rows])
            display, overlap = project_window_values(result["positions"], damage, result["token_grid"], args.window_size)
            np.savez_compressed(out / "maps" / f"query{query_id}.npz", token_attention=result["token_attention_map"],
                                positions=result["positions"], attention_score=np.array([r["attention_score"] for r in rows]),
                                margin_drop=damage, deletion_display=display, overlap_counts=overlap)
            collect_cases(rows, cases, counts, rng, args.cases_per_category)
            print(f"Queries {number+1}/{len(query_ids)} | grid={result['token_grid']} windows={len(rows)} "
                  f"batched forwards={result['variant_forward_calls']} rho={query['spearman']} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    summary = summarize_faithfulness(queries, seed=seed, resamples=args.bootstrap_samples, min_mean_spearman=args.min_mean_spearman)
    summary.update(n_windows=sum(q["n_windows"] for q in queries), reference_cache_reused=True,
                   max_clean_margin_delta_vs_source=max(abs(q["clean_margin_delta_vs_source"]) for q in queries))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    by_query = {q["query_id"]: q for q in queries}
    for query_id in query_ids[:3]:
        rgb, _ = dataset[dataset.num_references+query_id]
        with np.load(out / "maps" / f"query{query_id}.npz") as maps:
            plot_query(rgb, maps, by_query[query_id], out / "visualizations" / f"query{query_id}.png", args.window_size)
    for category, entries in cases.items():
        counts[category]["saved"] = len(entries)
        for entry in entries:
            query_id = entry["query_id"]
            prefix = out / "visualizations/cases" / f"{category}_query{query_id}"
            rgb, _ = dataset[dataset.num_references+query_id]
            with np.load(out / "maps" / f"query{query_id}.npz") as maps:
                plot_query(rgb, maps, by_query[query_id], prefix.with_suffix(".png"), args.window_size, selected=entry)
            prefix.with_suffix(".json").write_text(json.dumps(entry, indent=2)+"\n")
    (out / "visualizations/cases/categories.json").write_text(json.dumps(counts, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for ax, values, label in zip(axes, [[q["spearman"] for q in queries if q["spearman"] is not None],
                                       [q["top_minus_bottom_damage"] for q in queries],
                                       [q["attention_max_damage_percentile"] for q in queries]],
                                ["Within-query Spearman", "Top − bottom attention damage", "Attention-max damage percentile"]):
        ax.hist(values, bins=20, color="#2878b5", alpha=.8)
        ax.set(xlabel=label, ylabel="Queries")
        if label != "Attention-max damage percentile":
            ax.axvline(0, color="black", ls="--")
    fig.savefig(out / "plots/query_faithfulness.png", dpi=160); plt.close(fig)
    if file_hash(cache) != cache_hash or file_hash(source_csv) != config["source_csv_sha256"]:
        raise ValueError("Reference cache or source CSV changed during probe")
    write_report(out, summary, queries, config, counts)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
