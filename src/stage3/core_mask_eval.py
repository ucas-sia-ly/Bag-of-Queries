"""Dev-only native Core mean-fill evaluation against frozen full SUPPORT.

Masks are never resized. Exact native-pixel translations form random controls;
only the resulting RGB images enter Stage2's uint8 resize/normalize pipeline.
"""

from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision

from src.analysis.perturb import NoLegalTranslation, make_shape_matched_random_mask, perturb_rgb
from src.stage2.retrieval import (
    GSVSplit, RetrievalContext, cache_identity, evaluation_transform, file_sha256,
    load_support_cache, validate_descriptors,
)
from .cohort import ROOT
from .vulnerability_export import validate_cohort, verify_frozen_code

TAUS = (.3, .5, .7)
ARMS = tuple(f"family_wc_{tau}" for tau in TAUS) + ("ellipse_6pct",)
SELECTION_RULE = "max full-map weighted coverage; max target precision; min area; min centroid distance; candidate_id ascending"
FREEZE_RULE = "among family arms with >=2 paired queries: max query coverage, max mean paired margin advantage, then higher threshold; never use p-values"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_core(candidates, threshold):
    """One SOURCE/place per arm, chosen before any BoQ response is observed."""
    if threshold not in TAUS:
        raise ValueError("Only predeclared dev thresholds 0.3/0.5/0.7")
    eligible = [row for row in candidates if row["thresholds"][str(threshold)]["passes"]]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (
        -row["metrics"]["vulnerability_weighted_coverage"], -row["metrics"]["target_precision"],
        row["metrics"]["area_fraction"], row["metrics"]["centroid_distance_normalized"], row["candidate_id"],
    ))


def placement_seed(seed, image_key, mask):
    digest = hashlib.sha256(str(seed).encode()+b"\0"+image_key.encode()+b"\0"+mask.cpu().numpy().tobytes()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def exact_controls(core, *, seed):
    """Reuse Stage2 uniform legal translations WITH replacement, exactly 5."""
    masks, offsets, alternatives = make_shape_matched_random_mask(core, repeats=5, seed=seed)
    coordinates = core.nonzero()
    for random, offset in zip(masks, offsets):
        if (offset == (0, 0) or not torch.equal(coordinates + coordinates.new_tensor(offset), random.nonzero())
                or int(random.sum()) != int(core.sum())):
            raise ValueError("Random control changed native pixel shape/area/topology")
    return masks, offsets, alternatives


def native_mean_fill(decoded, masks):
    """Native RGB mean fill, then explicit uint8 quantization before frozen resize.

    Quantization is shared by targeted/random; original unmasked bytes are exact.
    No mask interpolation, rotation, dilation, deformation or clipping occurs.
    """
    if decoded.dtype != torch.uint8 or decoded.ndim != 3 or decoded.shape[0] != 3:
        raise ValueError("Require original decoded uint8 RGB")
    rgb = decoded.float()/255
    requested = rgb.mean((-2, -1))
    variants = perturb_rgb(rgb, masks, operator="mean_fill").mul(255).round().clamp(0, 255).to(torch.uint8)
    fill = (requested*255).round().to(torch.uint8)
    return variants, requested.tolist(), fill.tolist()


@torch.no_grad()
def evaluate_core(model, bank, decoded, core, *, image_key, place_key, seed, cached_descriptor=None,
                  cached_margin=None):
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("BoQ must be frozen in eval mode")
    if bank.mode != "support" or core.shape != decoded.shape[-2:] or core.dtype != torch.bool or not core.any():
        raise ValueError("Require full SUPPORT and a nonempty original-size bool Core")
    transform = evaluation_transform((224,224))
    clean_descriptor, _ = model(transform(decoded).unsqueeze(0).to(bank.references.device))
    validate_descriptors(clean_descriptor, rows=1, dim=bank.references.shape[1])
    clean = bank.query(clean_descriptor, place_key)[0]
    descriptor_error = float((clean_descriptor.cpu()-cached_descriptor).abs().max()) if cached_descriptor is not None else 0.
    margin_error = abs(clean["margin"]-cached_margin) if cached_margin is not None else 0.
    if descriptor_error > 1e-6 or margin_error > 1e-6:
        raise ValueError("Clean forward differs from frozen Stage3 vulnerability export")
    random_seed = placement_seed(seed,image_key,core)
    try:
        random,offsets,alternatives = exact_controls(core,seed=random_seed)
        eligible,reason = True,""
    except NoLegalTranslation as exc:
        random = core.new_empty((0,*core.shape))
        offsets,alternatives,eligible,reason = [],0,False,str(exc)
    masks = torch.cat((core[None],random))
    variants,requested,quantized = native_mean_fill(decoded,masks)
    # Keep exactly the frozen image preprocessing and checkpoint forward path.
    descriptors,_ = model(transform(variants).to(bank.references.device))
    metrics = bank.query(descriptors,place_key,clean_descriptor=clean_descriptor)
    base = dict(image_key=image_key,place_key=place_key,clean_margin=clean["margin"],clean_rank=clean["rank"],
                mask_pixels=int(core.sum()),mask_area_fraction=float(core.float().mean()),
                paired_eligible=eligible,exclusion_reason=reason,legal_random_placements=alternatives,
                unique_random_placements=len(set(offsets)),placement_seed=random_seed,
                descriptor_max_abs_error=descriptor_error,clean_margin_abs_error=margin_error)

    def row(metric,condition,repeat,offset,applied):
        return dict(base,condition=condition,random_repeat=repeat,shift_y=offset[0],shift_x=offset[1],
                    applied_mask_pixels=applied,perturbed_margin=metric["margin"],
                    margin_drop=clean["margin"]-metric["margin"],positive_sim=metric["positive_sim"],
                    negative_sim=metric["negative_sim"],perturbed_rank=metric["rank"],
                    descriptor_drift=metric["descriptor_drift"],hit_at_1=metric["hit_at_1"])

    rows = [row(clean,"clean",-1,(0,0),0),row(metrics[0],"targeted",-1,(0,0),int(core.sum()))]
    rows += [row(metric,"random",i,offset,int(core.sum())) for i,(metric,offset) in enumerate(zip(metrics[1:],offsets))]
    audit = dict(placement_seed=random_seed,offsets_dy_dx=offsets,legal_alternatives=alternatives,
                 exact_translation_verified=eligible,requested_mean_rgb_0_1=requested,applied_mean_rgb_uint8=quantized,
                 quantization="round to nearest uint8 before unchanged Stage2 uint8 bicubic resize",
                 native_shape=list(core.shape),descriptor_max_abs_error=descriptor_error,clean_margin_abs_error=margin_error)
    return rows, masks.cpu().numpy(), audit


def pair_queries(rows):
    """Collapse 5 draws INSIDE each query. Never count repeats as independent n."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"],row["image_key"])].append(row)
    pairs = []
    for (arm,image_key),group in sorted(groups.items()):
        clean = [r for r in group if r["condition"]=="clean"]
        target = [r for r in group if r["condition"]=="targeted"]
        random = [r for r in group if r["condition"]=="random"]
        if len(clean)!=1 or len(target)!=1:
            raise ValueError("Exactly one clean and targeted row required per query/arm")
        if len(group)!=len(clean)+len(target)+len(random):
            raise ValueError("Unknown condition in query")
        t = target[0]
        if not t["paired_eligible"]:
            if random: raise ValueError("Ineligible query has random rows")
            continue
        if sorted(r["random_repeat"] for r in random)!=list(range(5)):
            raise ValueError("Every eligible query requires exactly five random repeats")
        if any(r["mask_pixels"]!=t["mask_pixels"] or r["place_key"]!=t["place_key"]
               or r["clean_margin"]!=t["clean_margin"] for r in group):
            raise ValueError("Shape budget/place mismatch within query")
        if any(r["applied_mask_pixels"]!=t["mask_pixels"] for r in [t,*random]):
            raise ValueError("Intervention pixel count changed within query")
        mean = float(np.mean([r["margin_drop"] for r in random]))
        pairs.append(dict(arm=arm,image_key=image_key,place_key=t["place_key"],candidate_id=t["candidate_id"],
                          family=t["family"],target_margin_drop=t["margin_drop"],random_mean_margin_drop=mean,
                          paired_margin_difference=t["margin_drop"]-mean,random_repeats=5,
                          target_area_fraction=t["mask_area_fraction"]))
    return pairs


def summarize_pairs(pairs, arm, *, seed, stats_module):
    group = [r for r in pairs if r["arm"]==arm]
    target = np.array([r["target_margin_drop"] for r in group])
    random = np.array([r["random_mean_margin_drop"] for r in group])
    if len({r["place_key"] for r in group})!=len(group):
        raise ValueError("Statistics require one query per place")
    result = dict(arm=arm,n_queries=len(group),inference_scope="dev_selection_only_not_final_paper_test",
                  mean_target_margin_drop=float(target.mean()) if len(group) else None,
                  median_target_margin_drop=float(np.median(target)) if len(group) else None,
                  mean_random_margin_drop=float(random.mean()) if len(group) else None,
                  median_random_margin_drop=float(np.median(random)) if len(group) else None,
                  mean_paired_difference=float((target-random).mean()) if len(group) else None,
                  median_paired_difference=float(np.median(target-random)) if len(group) else None,
                  ci_low=None,ci_high=None,wilcoxon_statistic=None,wilcoxon_p_raw=None,
                  win_rate=float((np.round(target-random,12)>0).mean()) if len(group) else None,
                  tie_rate=float((np.round(target-random,12)==0).mean()) if len(group) else None,
                  bootstrap_resamples=20000,bootstrap_seed=seed,bootstrap_unit="query",
                  status="no_paired_queries" if not group else "insufficient_queries")
    if len(group)>=2:
        stat = stats_module.paired_statistics(target,random,seed=seed,resamples=20000,key=f"stage3-core:{arm}")
        result.update(status="descriptive_dev_statistics",ci_low=stat["ci_low"],ci_high=stat["ci_high"],
                      wilcoxon_statistic=stat["wilcoxon_statistic"],wilcoxon_p_raw=stat["wilcoxon_p_raw"],
                      win_rate=stat["left_win_rate"],tie_rate=stat["tie_rate"])
    return result


def freeze_choice(summaries):
    eligible = [r for r in summaries if r["arm"].startswith("family_wc_") and r["n_queries"]>=2]
    return max(eligible,key=lambda r:(r["n_queries"],r["mean_paired_difference"],float(r["arm"].split("_")[-1]))) if eligible else None


def _load_runtime(root, vulnerability_report):
    prior = _json(root/"outputs/stage2/retrieval/validation.json")
    if prior["status"]!="GO": raise ValueError("Frozen retrieval report must be GO")
    identity = prior["identity"]
    hashes = dict(identity["source_sha256"])
    config = _json(root/"outputs/stage2/gsv_occlusion/support/config.json")
    hashes.update(config["source_sha256"])
    audit = {name:verify_frozen_code(root,name,digest) for name,digest in hashes.items()}
    if identity!=vulnerability_report["reference_definition"]["stage2_retrieval_identity"]:
        raise ValueError("Vulnerability and frozen retrieval reference identities disagree")
    os.environ["XFORMERS_DISABLED"]="1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    split = GSVSplit.read(root/"outputs/stage2/split/gsv_split.jsonl")
    loader = _module(root/"scripts/visualize_attention.py","stage3_frozen_boq_loader")
    model,model_config = loader.load_model(prior["checkpoint_path"],identity["model"]["backbone"],identity["device"])
    current = cache_identity(split,identity["images_root"],prior["checkpoint_path"],model_config,
                             identity["image_size"],identity["batch_size"],identity["device"])
    if current["source_sha256"]!={name:audit[name]["current_sha256"] for name in identity["source_sha256"]}:
        raise ValueError("Runtime producer identity changed")
    compatible = dict(current,source_sha256=identity["source_sha256"])
    if compatible!=identity or file_sha256(prior["cache_path"])!=prior["cache_sha256"]:
        raise ValueError("Full SUPPORT cache/checkpoint/environment is not the frozen Stage2 context")
    descriptors = load_support_cache(prior["cache_path"],split,identity)
    bank = RetrievalContext.from_support(descriptors,split.support,mode="support",device=identity["device"])
    return model,bank,split,dict(retrieval_identity=current,producer_code_audit=audit,checkpoint_sha256=identity["checkpoint_sha256"],
                               cache_sha256=prior["cache_sha256"],reference_shape=list(bank.references.shape))


def run_core_mask_eval(*, repo_root=ROOT, adaptvpr_root=None, candidate_dir=None,
                       output_dir="outputs/stage3/dev/core_mask_eval", seed=0):
    root = Path(repo_root).resolve()
    adapt = Path(adaptvpr_root or root.parent/"workspace/AdaptVPR").resolve()
    candidates_path = Path(candidate_dir or adapt/"outputs/stage3_dev/candidate_masks").resolve()
    output = (root/output_dir).resolve()
    if output.exists() or not output.is_relative_to(root/"outputs/stage3/dev"):
        raise ValueError("Require a new output directory under outputs/stage3/dev")
    if type(seed) is not int or seed<0: raise ValueError("seed must be nonnegative")
    # Only numerical candidate helpers and the old ellipse adapter are imported.
    sys.path.append(str(adapt))
    from targeted.candidate_masks import CoverageContext, assess_thresholds
    from targeted.family_constraints import validate_constraints,config_digest
    from targeted.mask_adapter import adapt_generation_mask

    dev = root/"outputs/stage3/dev"
    cohort,cohort_summary = _jsonl(dev/"cohort.jsonl"),_json(dev/"summary.json")
    vulnerability_dir = dev/"vulnerability"
    vulnerability_report = _json(vulnerability_dir/"summary.json")
    vulnerability = {r["image_key"]:r for r in _jsonl(vulnerability_dir/"vulnerability.jsonl")}
    candidate_summary = _json(candidates_path/"summary.json")
    samples = _json(candidates_path/"samples.json")
    candidates = _jsonl(candidates_path/"candidates.jsonl")
    if (candidate_summary["status"]!="COMPLETE" or len(candidates)!=candidate_summary["candidate_count"]
            or file_sha256(candidates_path/"candidates.jsonl")!=candidate_summary["candidates_sha256"]
            or file_sha256(vulnerability_dir/"vulnerability.jsonl")!=vulnerability_report["records_sha256"]
            or file_sha256(dev/"cohort.jsonl")!=cohort_summary["cohort_sha256"]):
        raise ValueError("Incomplete or changed candidate/vulnerability/cohort artifacts")
    if len(cohort)!=50 or vulnerability_report["record_count"]!=50 or len(vulnerability)!=50:
        raise ValueError("Require complete dev-50 vulnerability input")
    config = validate_constraints(candidate_summary["configuration"])
    if config_digest(config)!=candidate_summary["configuration_sha256"] or config["tau_target_precision"]!=.7:
        raise ValueError("Require unchanged candidate configuration and target precision 0.7")
    input_hashes = dict(vulnerability_report["inputs_sha256"])
    for path,digest in candidate_summary["inputs_sha256"].items():
        if path in input_hashes and input_hashes[path]!=digest: raise ValueError("Conflicting input provenance")
        input_hashes[path]=digest
    for path in (dev/"cohort.jsonl",dev/"summary.json",vulnerability_dir/"summary.json",vulnerability_dir/"vulnerability.jsonl",
                 candidates_path/"summary.json",candidates_path/"samples.json",candidates_path/"candidates.jsonl"):
        if str(path) in input_hashes and input_hashes[str(path)]!=file_sha256(path): raise ValueError("Input hash mismatch")
        input_hashes[str(path)]=file_sha256(path)
    for path,digest in input_hashes.items():
        if file_sha256(path)!=digest: raise ValueError(f"Changed frozen input: {path}")
    for name,digest in candidate_summary["code_sha256"].items():
        if file_sha256(adapt/name)!=digest: raise ValueError(f"Candidate producer changed: {name}")
        input_hashes[str(adapt/name)]=digest
    for path in (adapt/"targeted/mask_adapter.py",root/"scripts/analyze_stage1_results.py"):
        input_hashes[str(path)]=file_sha256(path)
    if [s["target"]["image_key"] for s in samples]!=[r["image_key"] for r in cohort[:len(samples)]]:
        raise ValueError("Candidate sample subset/order differs from frozen dev")
    by_query = defaultdict(list)
    for row in candidates: by_query[row["image_key"]].append(row)
    # Persist every selection/no-selection before loading the model or observing retrieval.
    selections,exclusions,jobs = [],[],[]
    output.mkdir(parents=True)
    (output/"masks").mkdir()
    protocol = dict(seed=seed,random_repeats=5,bootstrap_resamples=20000,arms=list(ARMS),
                    candidate_selection_rule=SELECTION_RULE,dev_freeze_rule=FREEZE_RULE,
                    intervention_space="native SOURCE pixels; masks never resized",
                    rgb_preprocessing="native float RGB mean-fill; round to uint8; frozen Stage2 uint8 bicubic antialias resize to 224 and ImageNet normalization",
                    random_control="uniform integer translations with replacement; exclude (0,0); overlap with target allowed; same shape/area/topology",
                    primary_metric="retrieval margin drop",statistical_unit="query; one SOURCE/place per arm",
                    inference_scope="DEV_SELECTION_ONLY; p-values are descriptive, not final paper tests",
                    configuration=config,configuration_sha256=config_digest(config),inputs_sha256=input_hashes,
                    baseline="original AdaptVPR ellipse adapter: target_ratio=.06, min_overlap=.70; unchanged defaults; no weighted coverage filter",
                    ellipse_adapter_sha256=file_sha256(adapt/"targeted/mask_adapter.py"))
    _write_json(output/"protocol.json",protocol)
    for index,sample in enumerate(samples):
        key = sample["target"]["image_key"]
        original = cohort[index]
        v = vulnerability[key]
        if any(sample["target"][field]!=original[field] for field in ("image_key","place_key","source_sha256","source_identity","source_path")):
            raise ValueError("Candidate/source identity mismatch")
        artifact = (vulnerability_dir/v["numerical_artifact"]).resolve()
        if file_sha256(artifact)!=v["numerical_artifact_sha256"]: raise ValueError("Continuous vulnerability artifact changed")
        if file_sha256(original["source_path"])!=original["source_sha256"]: raise ValueError("SOURCE bytes changed")
        with np.load(artifact,allow_pickle=False) as arrays:
            roi,weights = arrays["attention_roi_token_mask"].copy(),arrays[config["weight_map"]].copy()
            if str(arrays["image_key"])!=key or str(arrays["place_key"])!=original["place_key"]:
                raise ValueError("NPZ source identity mismatch")
        context = CoverageContext(roi,weights,(v["source_height"],v["source_width"]))
        for tau in TAUS:
            arm = f"family_wc_{tau}"
            chosen = choose_core(by_query[key],tau)
            if chosen is None:
                exclusions.append(dict(arm=arm,image_key=key,place_key=original["place_key"],reason="no_candidate_passes_predeclared_gates"))
                continue
            path = (candidates_path/chosen["core_mask_path"]).resolve()
            if not path.is_relative_to(candidates_path) or file_sha256(path)!=chosen["core_mask_sha256"]:
                raise ValueError("Chosen Core mask path/hash mismatch")
            input_hashes[str(path)]=chosen["core_mask_sha256"]
            with Image.open(path) as image:
                raw = np.asarray(image)
            if raw.dtype!=np.uint8 or not np.isin(raw,[0,255]).all(): raise ValueError("Core must be binary 0/255 PNG")
            mask = raw==255
            metrics,_ = context.measure(mask)
            for name in ("target_precision","binary_roi_coverage","vulnerability_weighted_coverage","area_fraction","centroid_distance_normalized","num_components"):
                if not np.isclose(metrics[name],chosen["metrics"][name],atol=1e-12,rtol=0): raise ValueError("Chosen Core diagnostics disagree with pixels")
            gates = assess_thresholds(metrics,config["families"][chosen["family"]],chosen["geometry"],config)
            if not gates[str(tau)]["passes"]: raise ValueError("Chosen Core fails independently recomputed gates")
            selection = dict(arm=arm,image_key=key,place_key=original["place_key"],candidate_id=chosen["candidate_id"],
                             family=chosen["family"],weighted_coverage_threshold=tau,metrics=metrics,source_core_sha256=file_sha256(path))
            jobs.append((selection,mask,original,v,artifact))
        ellipse,diagnostic = adapt_generation_mask(Image.fromarray(context.roi_pixels.astype(np.uint8)),target_ratio=.06,min_overlap=.70)
        _write_json(output/f"ellipse_{index:02d}_diagnostic.json",dict(image_key=key,diagnostics=diagnostic))
        if ellipse is None:
            exclusions.append(dict(arm="ellipse_6pct",image_key=key,place_key=original["place_key"],reason=diagnostic["failure_reason"]))
        else:
            mask = np.asarray(ellipse)>0
            metrics,_ = context.measure(mask)
            selection = dict(arm="ellipse_6pct",image_key=key,place_key=original["place_key"],candidate_id=f"ellipse-{index:02d}",
                             family="ellipse_baseline",weighted_coverage_threshold=None,metrics=metrics)
            jobs.append((selection,mask,original,v,artifact))
        print(f"Prepared {index+1}/{len(samples)}: {key.split('/')[0]}",flush=True)
    for selection,mask,*_ in jobs:
        name = hashlib.sha256((selection["arm"]+selection["image_key"]).encode()).hexdigest()[:24]
        path = output/"masks"/f"{name}_target.png"
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        selection.update(saved_core_mask_path=path.relative_to(output).as_posix(),saved_core_mask_sha256=file_sha256(path))
        selections.append(selection)
    _write_json(output/"chosen_core_masks.json",selections)
    # Pin selected pixel artifacts before any model response is observed.
    _write_json(output/"protocol.json",protocol)
    _write_json(output/"exclusions.json",exclusions)
    model,bank,split,runtime = _load_runtime(root,vulnerability_report)
    excluded_places = set()
    for path in {r["config"] for r in cohort_summary["stage2_audit"]["cohorts"]}:
        excluded_places.update(r["place_key"] for records in _json(path)["cohorts"].values() for r in records)
    validate_cohort(cohort,cohort_summary,split,excluded_places)
    all_rows,translation_audits = [],[]
    for index,(selection,mask,source,v,artifact) in enumerate(jobs):
        decoded = torchvision.io.decode_image(source["source_path"],mode="RGB")
        with np.load(artifact,allow_pickle=False) as arrays:
            cached = torch.from_numpy(arrays["clean_descriptor"].copy())[None]
            margin = float(arrays["clean_margin"])
        rows,masks,audit = evaluate_core(model,bank,decoded,torch.from_numpy(mask),image_key=source["image_key"],
                                        place_key=source["place_key"],seed=seed,cached_descriptor=cached,cached_margin=margin)
        for row in rows: row.update(arm=selection["arm"],candidate_id=selection["candidate_id"],family=selection["family"])
        all_rows.extend(rows)
        path = output/"masks"/(Path(selection["saved_core_mask_path"]).stem.replace("_target","")+"_placements.npz")
        np.savez_compressed(path,masks=masks,offsets_dy_dx=np.array([(0,0),*audit["offsets_dy_dx"]],dtype=np.int64),
                            image_key=np.array(source["image_key"]),place_key=np.array(source["place_key"]))
        audit.update(arm=selection["arm"],image_key=source["image_key"],candidate_id=selection["candidate_id"],
                     placements_path=path.relative_to(output).as_posix(),placements_sha256=file_sha256(path))
        translation_audits.append(audit)
        if not rows[0]["paired_eligible"]:
            exclusions.append(dict(arm=selection["arm"],image_key=source["image_key"],place_key=source["place_key"],reason=rows[0]["exclusion_reason"]))
        print(f"Evaluated {index+1}/{len(jobs)} {selection['arm']} {source['place_key']}",flush=True)
    pairs = pair_queries(all_rows)
    stats_module = _module(root/"scripts/analyze_stage1_results.py","stage3_frozen_query_statistics")
    summaries = [summarize_pairs(pairs,arm,seed=seed,stats_module=stats_module) for arm in ARMS]
    pair_fields = ["arm","image_key","place_key","candidate_id","family","target_margin_drop","random_mean_margin_drop","paired_margin_difference","random_repeats","target_area_fraction"]
    row_fields = list(all_rows[0]) if all_rows else ["arm","image_key","place_key","condition","random_repeat","margin_drop"]
    for arm in (None,*ARMS):
        directory = output if arm is None else output/arm
        directory.mkdir(exist_ok=True)
        _write_csv(directory/"per_condition.csv",[r for r in all_rows if arm is None or r["arm"]==arm],row_fields)
        _write_csv(directory/"paired.csv",[r for r in pairs if arm is None or r["arm"]==arm],pair_fields)
    _write_csv(output/"statistics.csv",summaries,list(summaries[0]))
    # Baseline comparisons use only common SOURCE identities; never unpaired subset means.
    baseline = {r["image_key"]:r for r in pairs if r["arm"]=="ellipse_6pct"}
    common,common_stats = [],[]
    for arm in ARMS[:-1]:
        comparable = []
        for row in pairs:
            if row["arm"]!=arm or row["image_key"] not in baseline: continue
            b = baseline[row["image_key"]]
            entry = dict(arm=arm,image_key=row["image_key"],place_key=row["place_key"],
                         family_margin_advantage=row["paired_margin_difference"],ellipse_margin_advantage=b["paired_margin_difference"],
                         difference=row["paired_margin_difference"]-b["paired_margin_difference"])
            common.append(entry)
            comparable.append(dict(entry,target_margin_drop=entry["family_margin_advantage"],random_mean_margin_drop=entry["ellipse_margin_advantage"]))
        common_stats.append(summarize_pairs(comparable,arm,seed=seed,stats_module=stats_module))
    _write_csv(output/"ellipse_common_queries.csv",common,["arm","image_key","place_key","family_margin_advantage","ellipse_margin_advantage","difference"])
    _write_json(output/"ellipse_common_statistics.json",dict(contrast="family target-minus-own-random advantage minus ellipse target-minus-own-random advantage on common queries",statistics=common_stats))
    chosen = freeze_choice(summaries)
    frozen = dict(schema_version=1,status="FROZEN_DEV_CONFIGURATION" if chosen else "NO_ELIGIBLE_CONFIGURATION",
                  selected_weighted_coverage_threshold=float(chosen["arm"].split("_")[-1]) if chosen else None,
                  candidate_configuration=config,candidate_selection_rule=SELECTION_RULE,freeze_rule=FREEZE_RULE,
                  inference_scope="dev selection only; no final-paper significance claim",selected_dev_statistics=chosen,
                  random_repeats=5,seed=seed,checkpoint_sha256=runtime["checkpoint_sha256"],
                  support_cache_sha256=runtime["cache_sha256"],intervention_protocol=protocol["rgb_preprocessing"],
                  candidate_config_sha256=config_digest(config),protocol_sha256=file_sha256(output/"protocol.json"))
    _write_json(output/"frozen_config.json",frozen)
    _write_json(output/"random_placements.json",translation_audits)
    _write_json(output/"exclusions.json",exclusions)
    for path,digest in input_hashes.items():
        if file_sha256(path)!=digest: raise ValueError(f"Input changed during evaluation: {path}")
    report = dict(status="COMPLETE",inference_scope="DEV_SELECTION_ONLY_NOT_FINAL_PAPER_TEST",dev_cohort_size=50,
                  evaluated_candidate_subset=len(samples),remaining_dev_not_evaluated=50-len(samples),
                  chosen_masks=len(jobs),per_condition_rows=len(all_rows),paired_rows=len(pairs),statistics=summaries,
                  runtime=runtime,candidate_selection_rule=SELECTION_RULE,freeze_rule=FREEZE_RULE,
                  selected_weighted_coverage_threshold=frozen["selected_weighted_coverage_threshold"],
                  random_repeats=5,bootstrap_resamples=20000,query_averaging_before_statistics=True,
                  maximum_clean_descriptor_error=max((r["descriptor_max_abs_error"] for r in all_rows),default=0),
                  input_files_unchanged=True,diffusion_called=False,training_performed=False,
                  code_sha256={name:file_sha256(root/name) for name in ("src/stage3/core_mask_eval.py","scripts/stage3_eval_core_masks.py")})
    _write_json(output/"summary.json",report)
    return report
