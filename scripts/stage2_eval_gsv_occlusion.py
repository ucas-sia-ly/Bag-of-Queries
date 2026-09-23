"""Evaluate fixed-budget Attention/Fused vs shape-matched Random on GSV."""

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2 as T

from src.stage2.gsv_occlusion import evaluate_query, sample_unique_places, summarize_experiment
from src.stage2.retrieval import ROOT, GSVSplit, RetrievalContext, cache_identity, file_sha256, load_support_cache
from src.stage2.vulnerability import VulnerabilityEstimator


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def plot_diagnostics(pairs, seeds, output_dir):
    for seed in seeds:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for row, strategy in enumerate(("attention", "fused")):
            group = [p for p in pairs if p["seed"] == seed and p["strategy"] == strategy]
            if not group:
                for axis in axes[row]:
                    axis.text(.5, .5, "No eligible pairs", ha="center")
                continue
            targeted = np.array([p["targeted_margin_drop"] for p in group])
            random = np.array([p["random_margin_drop"] for p in group])
            lower, upper = min(targeted.min(), random.min(), 0.), max(targeted.max(), random.max(), .01)
            axes[row, 0].scatter(random, targeted, s=23, alpha=.7)
            axes[row, 0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
            axes[row, 0].set(xlabel="Random mean margin drop", ylabel="Targeted margin drop",
                             title=f"{strategy}: paired queries (n={len(group)})")
            for values, label in ((random, "Random mean"), (targeted, "Targeted")):
                axes[row, 1].step(np.sort(values), np.arange(1, len(values) + 1)/len(values), where="post", label=label)
            axes[row, 1].set(xlabel="Margin drop", ylabel="Empirical CDF", title="Query-level distributions")
            axes[row, 1].legend()
            axes[row, 2].boxplot([np.zeros(len(group)), random, targeted], tick_labels=["Clean", "Random mean", "Targeted"])
            axes[row, 2].axhline(0, color="grey", linewidth=.7)
            axes[row, 2].set(ylabel="Margin drop", title="Equal-shape / equal-area occlusion")
        fig.suptitle(f"GSV | seed={seed} | one SOURCE per place | random draws averaged within query")
        fig.savefig(output_dir / f"diagnostics_seed{seed}.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, default=ROOT / "outputs/stage2/retrieval/validation.json")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/stage2/split/gsv_split.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stage2/gsv_occlusion/support")
    parser.add_argument("--reference-mode", choices=["support", "prototype"], default="support")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--mask-ratio", type=float, default=.15)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if (args.num_queries < 2 or args.random_repeats < 1 or args.bootstrap_resamples < 10000
            or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds)):
        parser.error("Require >=2 queries, positive repeats, >=10,000 resamples, distinct nonnegative seeds")
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from scripts.visualize_attention import load_model

    prior = json.loads(args.retrieval_report.read_text())
    if prior["status"] != "GO":
        raise ValueError("Retrieval must pass validation before this experiment")
    split = GSVSplit.read(args.split)
    cohorts = {seed: sample_unique_places(split, args.num_queries, seed) for seed in args.seeds}
    identity = prior["identity"]
    device = identity["device"]
    model, model_config = load_model(prior["checkpoint_path"], identity["model"]["backbone"], device)
    current = cache_identity(split, identity["images_root"], prior["checkpoint_path"], model_config,
                             identity["image_size"], identity["batch_size"], device)
    descriptors = load_support_cache(prior["cache_path"], split, current)
    if current != identity or file_sha256(prior["cache_path"]) != prior["cache_sha256"]:
        raise ValueError("Current model/data/cache does not match the validated retrieval context")
    bank = RetrievalContext.from_support(descriptors, split.support, mode=args.reference_mode, device=device)
    del descriptors
    estimator = VulnerabilityEstimator(top_k=args.top_k, alpha=args.alpha, mask_ratio=args.mask_ratio,
                                       batch_size=args.batch_size, mode="fused", strict=False)
    transform = T.Compose([T.Resize(tuple(identity["image_size"]), interpolation=T.InterpolationMode.BICUBIC,
                                     antialias=True), T.ToDtype(torch.float32, scale=True)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_files = ["scripts/stage2_eval_gsv_occlusion.py", "src/stage2/gsv_occlusion.py",
                    "src/stage2/vulnerability.py", "src/analysis/attention.py", "src/analysis/masks.py",
                    "src/analysis/perturb.py", "src/analysis/retrieval.py", "scripts/analyze_stage1_results.py"]
    config = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        retrieval_identity=identity, cache_sha256=prior["cache_sha256"],
        retrieval_report_sha256=file_sha256(args.retrieval_report), reference_shape=list(bank.references.shape),
        source_sha256={name: file_sha256(ROOT / name) for name in source_files},
        cohorts={str(seed): [asdict(record) for record in records] for seed, records in cohorts.items()},
        protocol=dict(primary="attention", supplementary="fused", random="uniform legal pixel translation with replacement",
                      seed_changes="SOURCE place/image sampling and random placements; split/support assignment fixed",
                      sampling="uniform hashed places, one hashed SOURCE per place",
                      fill="original RGB per-channel spatial mean; normalize after fill",
                      budget="round(ratio * tokens), equal pixel shape/area; random shifts need not align to tokens",
                      exclusions="ONLY impossible rigid translations, independently per strategy",
                      estimator_diagnostics="retained, never used to select or exclude queries",
                      gate="attention CI_low>0 and Holm Wilcoxon p<0.05 and >=80% pairs in EVERY seed",
                      generator_action="report decision only; no generator is run"),
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(dict(status="STOP", reason="evaluation not complete")) + "\n")
    rows, diagnostics = [], []
    for seed, records in cohorts.items():
        masks = {}
        for index, record in enumerate(records):
            rgb = transform(torchvision.io.decode_image(Path(identity["images_root"]) / record.image_key, mode="RGB")).to(device)
            values, diagnostic, target_masks = evaluate_query(model, rgb, record, bank, estimator,
                                                             seed=seed, repeats=args.random_repeats, batch_size=args.batch_size)
            rows.extend(values)
            diagnostics.append(diagnostic)
            masks.update({f"query_{index:03d}_{strategy}": mask.cpu().numpy() for strategy, mask in target_masks.items()})
            if (index + 1) % 10 == 0:
                print(f"{args.reference_mode} seed={seed}: {index + 1}/{len(records)} SOURCE images", flush=True)
        np.savez_compressed(args.output_dir / f"target_masks_seed{seed}.npz", **masks)
        write_csv(args.output_dir / "per_query.csv", rows)
        (args.output_dir / "estimator_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")
    analysis = summarize_experiment(rows, seeds=args.seeds, repeats=args.random_repeats, resamples=args.bootstrap_resamples)
    if any(c["n_total"] != args.num_queries for c in analysis["comparisons"]):
        raise ValueError("Incomplete preselected SOURCE cohort")
    write_csv(args.output_dir / "paired_queries.csv", analysis["pairs"])
    write_csv(args.output_dir / "group_summary.csv", analysis["summaries"])
    write_csv(args.output_dir / "paired_statistics.csv", analysis["comparisons"])
    (args.output_dir / "exclusions.json").write_text(json.dumps(analysis["exclusions"], indent=2) + "\n")
    plot_diagnostics(analysis["pairs"], args.seeds, args.output_dir)
    summary = {key: value for key, value in analysis.items() if key not in ("pairs", "summaries", "exclusions")}
    summary.update(reference_mode=args.reference_mode, num_raw_rows=len(rows),
                   num_estimator_stops=sum(d["status"] == "STOP" for d in diagnostics),
                   max_clean_descriptor_error=max(d["clean_batch_max_abs_error"] for d in diagnostics),
                   fused_effect_supported=all(c["supported"] for c in analysis["comparisons"] if c["strategy"] == "fused"))
    if len(args.seeds) < 2:
        summary.update(status="STOP", reason="Need >=2 seeds to pass the consistency gate")
    summary["fused_estimator_ready"] = summary["fused_effect_supported"] and summary["num_estimator_stops"] == 0
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "GO":
        raise SystemExit("STOP: GSV attention-targeted advantage is not supported consistently; do not start generation")


if __name__ == "__main__":
    main()
