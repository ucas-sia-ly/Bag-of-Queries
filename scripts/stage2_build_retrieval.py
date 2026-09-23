"""Encode all GSV SUPPORT images and audit 100 SOURCE queries against the trainer."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.stage2.retrieval import (
    ROOT, GSVImages, GSVSplit, RetrievalContext, cache_identity, encode_images,
    file_sha256, get_support_cache,
)


def trainer_recalls(references, queries, ground_truth, directory):
    """Call the actual trainer helper; disk backing avoids a second large RAM copy."""
    import faiss
    from src.utils import compute_recall_performance

    faiss.omp_set_num_threads(4)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        combined = np.memmap(Path(temporary) / "trainer.f32", mode="w+", dtype="float32",
                             shape=(len(references) + len(queries), references.shape[1]))
        for start in range(0, len(references), 2048):
            end = min(start + 2048, len(references))
            combined[start:end] = references[start:end].cpu().numpy()
        combined[len(references):] = queries.cpu().numpy()
        combined.flush()
        result = compute_recall_performance(combined, len(references), len(queries), ground_truth,
                                            k_values=[1, 5, 10])
        del combined
    return {f"R@{k}": float(value) for k, value in result.items()}


def validate_context(context, queries, records, output_dir):
    rows = [dict(image_key=record.image_key, place_key=record.place_key,
                 **context.query(query, record.place_key)[0]) for query, record in zip(queries, records)]
    recalls = {f"R@{k}": float(np.mean([row[f"hit_at_{k}"] for row in rows])) for k in (1, 5, 10)}
    trainer = trainer_recalls(context.references, queries,
                              [context.positive_indices(r.place_key) for r in records], output_dir)
    if recalls != trainer:
        raise ValueError(f"STOP: {context.mode} recalls disagree with original trainer: {recalls} vs {trainer}")
    # Independent double-precision cosine oracle for the first three real queries.
    # Block over references to keep RAM bounded; this is validation only, not the API.
    errors = []
    for query, record, row in zip(queries[:3], records[:3], rows[:3]):
        q = query.cpu().double().numpy()
        q /= np.linalg.norm(q)
        positives = set(context.positive_indices(record.place_key))
        best_positive, best_negative = -np.inf, -np.inf
        for start in range(0, len(context.references), 2048):
            reference = context.references[start:start + 2048].cpu().double().numpy()
            scores = reference @ q / np.linalg.norm(reference, axis=1)
            mask = np.array([start + i in positives for i in range(len(scores))])
            if mask.any():
                best_positive = max(best_positive, scores[mask].max())
            if (~mask).any():
                best_negative = max(best_negative, scores[~mask].max())
        error = max(abs(row["positive_sim"] - best_positive), abs(row["negative_sim"] - best_negative),
                    abs(row["margin"] - (best_positive - best_negative)))
        errors.append(float(error))
    if max(errors) > 2e-5:
        raise ValueError(f"STOP: cosine oracle mismatch: {errors}")
    return rows, {"recalls": recalls, "trainer_recalls": trainer,
                  "mean_margin": float(np.mean([r["margin"] for r in rows])),
                  "oracle_queries": len(errors), "oracle_max_abs_error": max(errors),
                  "oracle_tolerance": 2e-5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/stage2/split/gsv_split.jsonl")
    parser.add_argument("--images-root", type=Path, default=ROOT / "data/train/gsv-cities/Images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/stage2/gsv_support.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stage2/retrieval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true", help="Build cache without claiming the validation gate passed")
    args = parser.parse_args(argv)
    if min(args.image_size) <= 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid image size, batch size or workers")
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Import after disabling xformers, before loading the existing strict BoQ loader.
    from scripts.visualize_attention import load_model

    started = time.monotonic()
    split = GSVSplit.read(args.split)
    queries_records = split.sample_sources(args.num_queries, args.seed)
    model, model_config = load_model(args.checkpoint, args.backbone, args.device)
    patch_size = getattr(model.backbone, "patch_size", 1)
    if any(size % patch_size for size in args.image_size):
        raise ValueError("Image dimensions must be divisible by the backbone patch size")
    identity = cache_identity(split, args.images_root, args.checkpoint, model_config,
                              args.image_size, args.batch_size, args.device)
    print(f"SUPPORT={len(split.support)}, SOURCE={len(split.source)}, dim={model_config['descriptor_dim']}", flush=True)
    descriptors, reused = get_support_cache(args.cache, model, split, args.images_root, identity,
                                            device=args.device, workers=args.workers)
    print(f"Cache ready: {args.cache}; reused={reused}", flush=True)
    if args.cache_only:
        return
    queries = encode_images(model, GSVImages(queries_records, args.images_root, args.image_size),
                            device=args.device, batch_size=args.batch_size, workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "STOP", "identity": identity, "cache_path": str(args.cache.resolve()),
        "cache_sha256": file_sha256(args.cache), "cache_reused": reused,
        "checkpoint_path": str(args.checkpoint.resolve()), "support_shape": list(descriptors.shape),
        "num_queries": len(queries_records), "seed": args.seed,
        "query_sampling": "SHA256(seed + NUL + SOURCE image_key), first N",
        "query_image_keys": [r.image_key for r in queries_records],
        "validation_source_sha256": {name: file_sha256(ROOT / name) for name in
                                      ["scripts/stage2_build_retrieval.py", "src/analysis/retrieval.py", "src/utils.py"]},
        "prototype_recommendation_thresholds": {"spearman_min": .95, "sign_agreement_min": .95,
                                                   "max_recall_difference": .01},
    }
    # Retain a STOP marker if a later validation raises; never leave a stale GO report.
    report_path = args.output_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    results = {}
    for mode in ("support", "prototype"):
        print(f"Validating {mode} references against original trainer and cosine oracle", flush=True)
        context = RetrievalContext.from_support(descriptors, split.support, mode=mode, device=args.device)
        rows, metrics = validate_context(context, queries, queries_records, args.output_dir)
        results[mode] = rows
        report[mode] = dict(metrics, reference_shape=list(context.references.shape))
        if mode == "prototype":
            ids = np.random.default_rng(args.seed).choice(len(context.references), size=min(5, len(context.references)), replace=False)
            report["prototype_norm_checks"] = [
                dict(place_key=context.reference_place_keys[i],
                     norm=float(context.references[i].norm()), support_count=2) for i in ids
            ]
        with (args.output_dir / f"{mode}_queries.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        del context
    from scipy.stats import spearmanr
    a, b = (np.array([row["margin"] for row in results[mode]]) for mode in ("support", "prototype"))
    correlation = float(spearmanr(a, b).statistic) if len(a) > 1 and a.std() and b.std() else None
    sign_agreement = float(np.mean((a > 0) == (b > 0)))
    recall_difference = max(abs(report["support"]["recalls"][key] - report["prototype"]["recalls"][key])
                            for key in report["support"]["recalls"])
    report["comparison"] = dict(margin_spearman=correlation, margin_sign_agreement=sign_agreement,
                                 margin_mean_abs_difference=float(np.abs(a - b).mean()),
                                 max_recall_difference=recall_difference)
    report["recommended_reference_mode"] = (
        "prototype" if correlation is not None and correlation >= .95 and sign_agreement >= .95
        and recall_difference <= .01 else "support"
    )
    report.update(status="GO", elapsed_seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"GO: {report['support']['recalls']}; recommended={report['recommended_reference_mode']}", flush=True)


if __name__ == "__main__":
    main()
