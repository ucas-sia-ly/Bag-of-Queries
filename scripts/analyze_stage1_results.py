"""Formal query-paired statistics from saved stage-1 CSVs; no model imports.

Run with the project's scientific Python environment. All random placements are
averaged inside query before inference; bootstrap resamples queries, not draws.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


RATIOS = (.10, .15, .20)
MODES = {"connected_topk": "attention_connected", "raw_topk": "attention_raw_topk"}
BASELINES = ("random_token", "random_pixel")
ENDPOINTS = ("margin_drop", "descriptor_drift", "positive_similarity_drop",
             "rank_degradation", "failure_flip", "hit_at_1", "hit_at_5", "hit_at_10")
CONSTRUCTION = "attention_raw_topk-minus-attention_connected"
PRIMARY = "attention_connected-minus-random_token"
RAW = "attention_raw_topk-minus-own-random_token"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_frames(actual, expected, keys, numeric, labels=()):
    """Reject missing/extra/duplicate pairs and inconsistent saved aggregates."""
    require(not actual.duplicated(keys).any(), f"Duplicate rows: {keys}")
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    require(len(a) == len(e), f"Row count mismatch: {len(a)} != {len(e)}")
    require(a[list(keys) + list(labels)].equals(e[list(keys) + list(labels)]),
            f"Query/cohort labels differ: {keys}")
    require(np.isfinite(a[list(numeric)].to_numpy(float)).all(), "Non-finite saved endpoints")
    require(np.allclose(a[list(numeric)], e[list(numeric)], atol=1e-12, rtol=1e-10),
            "Saved paired endpoints disagree with within-query reconstruction")


def validate_inputs(per, pairs, ablations, expected_queries=500):
    """Reconstruct both paired files from per_query, checking all eight endpoints."""
    require(not per.duplicated(["query_id", "condition", "mask_mode", "mask_ratio", "random_repeat"]).any(),
            "Duplicate per-query draw")
    require(np.isfinite(per[list(ENDPOINTS)].to_numpy(float)).all(), "Non-finite per-query endpoints")
    clean = per[per.condition == "clean"].set_index("query_id").sort_index()
    require(clean.index.is_unique and len(clean) == expected_queries, "Unexpected clean query cohort")
    require(set(per.query_id) == set(clean.index), "Perturbations contain unknown queries")
    require(per.seed.nunique() == 1, "Mixed experiment seeds")
    require(set(per.condition) == {"clean", *MODES.values(), *BASELINES}, "Unexpected conditions")
    require(set(per.loc[per.condition != "clean", "mask_ratio"]) == set(RATIOS), "Unexpected ratios")
    require((per.clean_rank == per.query_id.map(clean.clean_rank)).all(), "Clean ranks differ across conditions")
    for k in (1, 5, 10):
        require((per[f"hit_at_{k}"] == (per.perturbed_rank <= k)).all(), f"Invalid R@{k}")
    require((per.failure_flip == ((per.clean_rank == 1) & (per.perturbed_rank != 1))).all(), "Invalid failure flips")
    for column, derived in {
        "rank_degradation": per.perturbed_rank - per.clean_rank,
        "margin_drop": per.clean_margin - per.perturbed_margin,
        "positive_similarity_drop": per.clean_positive_sim - per.perturbed_positive_sim,
        "clean_margin": per.clean_positive_sim - per.clean_negative_sim,
        "perturbed_margin": per.perturbed_positive_sim - per.perturbed_negative_sim,
    }.items():
        # Model metrics were calculated in float32 before CSV serialization.
        require(np.allclose(per[column], derived, atol=1e-7, rtol=1e-6), f"Invalid {column}")
    expected_pairs, expected_ablations, cohorts = [], [], []
    accounted = len(clean)
    for ratio in RATIOS:
        targets, random_means = {}, {}
        for mode, condition in MODES.items():
            target = per[(per.mask_ratio == ratio) & (per.mask_mode == mode) & (per.condition == condition)].set_index("query_id").sort_index()
            require(target.index.is_unique and target.index.equals(clean.index), "Target query cohort differs")
            targets[mode] = target
            accounted += len(target)
            for baseline in BASELINES:
                draws = per[(per.mask_ratio == ratio) & (per.mask_mode == mode) & (per.condition == baseline)]
                accounted += len(draws)
                eligible = target[f"eligible_{baseline}"]
                require(eligible.isin([True, False]).all(), "Invalid eligibility flag")
                ids = target.index[eligible]
                grouped = draws.groupby("query_id")
                require(set(grouped.groups) == set(ids), "Random draws disagree with eligibility")
                require(grouped.random_repeat.apply(lambda x: sorted(x) == list(range(5))).all(),
                        "Each eligible query must have exactly 5 placements numbered 0..4")
                require((draws.mask_pixels == draws.query_id.map(target.mask_pixels)).all(), "Unequal pixel budgets")
                if baseline == "random_token":
                    require((draws.mask_tokens == draws.query_id.map(target.mask_tokens)).all(), "Unequal token budgets")
                means = grouped[list(ENDPOINTS)].mean().sort_index()
                random_means[mode, baseline] = means
                require(mode != "connected_topk" or len(means) == expected_queries, "Incomplete primary cohort")
                cohorts.append(dict(mask_ratio=ratio, mask_mode=mode, baseline=baseline,
                                    n_total=len(target), n_paired=len(means), n_excluded=len(target)-len(means)))
                for q in means.index:
                    row = dict(query_id=q, mask_ratio=ratio, mask_mode=mode, baseline=baseline,
                               attention_condition=condition, clean_rank=int(target.loc[q, "clean_rank"]))
                    for endpoint in ENDPOINTS:
                        row[f"attention_{endpoint}"] = target.loc[q, endpoint]
                        row[f"random_{endpoint}"] = means.loc[q, endpoint]
                    row["paired_margin_difference"] = row["attention_margin_drop"] - row["random_margin_drop"]
                    expected_pairs.append(row)
        for budget in ("mask_tokens", "mask_pixels"):
            require((targets["raw_topk"][budget] == targets["connected_topk"][budget]).all(), "Construction budgets differ")
        contrasts = [(CONSTRUCTION, targets["raw_topk"], targets["connected_topk"])]
        contrasts += [(f"{m}:random_token-minus-random_pixel", random_means[m, "random_token"],
                       random_means[m, "random_pixel"]) for m in MODES]
        for name, left, right in contrasts:
            for q in left.index.intersection(right.index):
                row = dict(comparison=name, query_id=q, mask_ratio=ratio)
                for endpoint in ENDPOINTS:
                    row[f"left_{endpoint}"] = left.loc[q, endpoint]
                    row[f"right_{endpoint}"] = right.loc[q, endpoint]
                expected_ablations.append(row)
    require(accounted == len(per), "Unrecognized per-query rows")
    compare_frames(pairs, pd.DataFrame(expected_pairs), ["mask_mode", "baseline", "mask_ratio", "query_id"],
                   ["clean_rank", "paired_margin_difference"] + [f"{side}_{e}" for side in ("attention", "random") for e in ENDPOINTS],
                   ["attention_condition"])
    compare_frames(ablations, pd.DataFrame(expected_ablations), ["comparison", "mask_ratio", "query_id"],
                   [f"{side}_{e}" for side in ("left", "right") for e in ENDPOINTS])
    return clean, pd.DataFrame(cohorts)


def keyed_rng(seed, key):
    digest = hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, *np.frombuffer(digest[:16], dtype="<u4").tolist()]))


def bootstrap_mean_ci(values, *, resamples, seed, key, batch_size=256):
    """Percentile CI; a row is one query, columns share the same resampled IDs."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), "Invalid bootstrap data")
    require(resamples > 0 and batch_size > 0, "Invalid bootstrap size")
    rng = keyed_rng(seed, key)
    means = np.empty((resamples, x.shape[1]))
    for start in range(0, resamples, batch_size):
        indices = rng.integers(0, len(x), size=(min(batch_size, resamples-start), len(x)))
        means[start:start+len(indices)] = x[indices].mean(axis=1)
    return np.quantile(means, [.025, .975], axis=0)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid p-values")
    order = np.argsort(p, kind="stable")
    result = np.empty_like(p)
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def paired_statistics(left, right, *, seed, resamples, key):
    a, b = np.asarray(left, float), np.asarray(right, float)
    require(a.ndim == 1 and a.shape == b.shape and len(a) >= 2, "Need at least two paired queries")
    d = a - b
    require(np.isfinite(d).all(), "Non-finite paired differences")
    lo, hi = bootstrap_mean_ci(d, seed=seed, resamples=resamples, key=key)[:, 0]
    # Round only rank-test / win-tie inputs to avoid binary subtraction artifacts.
    ranked_d = np.round(d, decimals=12)
    nonzero = ranked_d[ranked_d != 0]
    if len(nonzero):
        w = stats.wilcoxon(ranked_d, zero_method="wilcox", alternative="two-sided", method="asymptotic")
        ranks = stats.rankdata(np.abs(nonzero), method="average")
        rank_biserial = np.sum(ranks * np.sign(nonzero)) / ranks.sum()
        wstat, wp = float(w.statistic), float(w.pvalue)
    else:
        wstat, wp, rank_biserial = 0., 1., 0.
    sd = float(d.std(ddof=1))
    if sd == 0:
        tstat = 0. if d.mean() == 0 else float(np.copysign(np.inf, d.mean()))
        tp, dz = (1., 0.) if d.mean() == 0 else (0., float("nan"))
    else:
        t = stats.ttest_rel(a, b, alternative="two-sided")
        tstat, tp, dz = float(t.statistic), float(t.pvalue), float(d.mean()/sd)
    return dict(n_queries=len(d), left_mean=float(a.mean()), right_mean=float(b.mean()),
                mean_paired_difference=float(d.mean()), median_paired_difference=float(np.median(d)),
                ci_low=float(lo), ci_high=float(hi), paired_sd=sd, cohen_dz=dz,
                rank_biserial=float(rank_biserial), wilcoxon_statistic=wstat, wilcoxon_p_raw=wp,
                t_statistic=tstat, t_df=len(d)-1, t_p_raw=tp,
                left_win_rate=float((ranked_d > 0).mean()), right_win_rate=float((ranked_d < 0).mean()),
                tie_rate=float((ranked_d == 0).mean()), bootstrap_resamples=resamples, bootstrap_seed=seed)


def margin_tables(pairs, ablations, args, total_queries):
    primary, secondary, protocol = [], [], []
    for ratio in RATIOS:
        for mode, name, destination in [("connected_topk", PRIMARY, primary), ("raw_topk", RAW, secondary)]:
            group = pairs[(pairs.mask_mode == mode) & (pairs.baseline == "random_token") & (pairs.mask_ratio == ratio)].sort_values("query_id")
            row = paired_statistics(group.attention_margin_drop, group.random_margin_drop,
                                    seed=args.seed, resamples=args.bootstrap_resamples, key=f"margin:{name}:{ratio}")
            row.update(comparison=name, mask_ratio=ratio, mask_mode=mode, baseline="random_token",
                       n_excluded=total_queries-len(group), attention_win_rate=row["left_win_rate"],
                       random_win_rate=row["right_win_rate"])
            destination.append(row)
        for name in sorted(ablations.comparison.unique()):
            group = ablations[(ablations.comparison == name) & (ablations.mask_ratio == ratio)].sort_values("query_id")
            row = paired_statistics(group.left_margin_drop, group.right_margin_drop,
                                    seed=args.seed, resamples=args.bootstrap_resamples, key=f"margin:{name}:{ratio}")
            row.update(comparison=name, mask_ratio=ratio, n_excluded=total_queries-len(group))
            row["interpretation"] = ("no stable difference detected" if row["ci_low"] <= 0 <= row["ci_high"]
                                     else "positive difference detected" if row["ci_low"] > 0 else "negative difference detected")
            (secondary if name == CONSTRUCTION else protocol).append(row)
    primary = pd.DataFrame(primary)
    for test in ("wilcoxon", "t"):
        primary[f"{test}_p_holm"] = holm(primary[f"{test}_p_raw"])
    primary["p_adjustment_family"] = "three primary ratios separately for each test; Wilcoxon primary, t sensitivity"
    secondary = pd.DataFrame(secondary)
    # Secondary claims are exploratory; also provide Holm over all six contrasts.
    for test in ("wilcoxon", "t"):
        secondary[f"{test}_p_holm"] = holm(secondary[f"{test}_p_raw"])
    secondary["p_adjustment_family"] = "six secondary contrasts separately for each test; exploratory"
    return primary, secondary, pd.DataFrame(protocol)


def recall_table(pairs, args):
    rows = []
    for ratio in RATIOS:
        all_queries = pairs[(pairs.mask_mode == "connected_topk") & (pairs.baseline == "random_token") & (pairs.mask_ratio == ratio)].sort_values("query_id")
        for subset in ("all_queries", "clean_R1_correct"):
            group = all_queries if subset == "all_queries" else all_queries[all_queries.clean_rank == 1]
            metrics = ("hit_at_1", "hit_at_5", "hit_at_10", "failure_flip")
            advantages = np.column_stack([(group[f"random_{m}"] - group[f"attention_{m}"]) * (-1 if m == "failure_flip" else 1) for m in metrics])
            ci = bootstrap_mean_ci(advantages, resamples=args.bootstrap_resamples, seed=args.seed, key=f"recall:{ratio}:{subset}")
            for j, metric in enumerate(metrics):
                rows.append(dict(comparison=PRIMARY, mask_ratio=ratio, subset=subset, metric=metric,
                                 n_queries=len(group), attention_mean=group[f"attention_{metric}"].mean(),
                                 random_token_mean=group[f"random_{metric}"].mean(),
                                 clean_mean=0. if metric == "failure_flip" else (group.clean_rank <= int(metric.split("_")[-1])).mean(),
                                 damage_advantage=advantages[:, j].mean(), ci_low=ci[0, j], ci_high=ci[1, j],
                                 definition="attention-minus-random" if metric == "failure_flip" else "random-minus-attention",
                                 bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.seed))
    return pd.DataFrame(rows)


def rank_table(pairs, thresholds):
    rows = []
    for ratio in RATIOS:
        group = pairs[(pairs.mask_mode == "connected_topk") & (pairs.baseline == "random_token") & (pairs.mask_ratio == ratio)]
        for side in ("attention", "random"):
            x = group[f"{side}_rank_degradation"].to_numpy()
            row = dict(mask_ratio=ratio, condition="attention_connected" if side == "attention" else "random_token",
                       n_queries=len(x), minimum=x.min(), maximum=x.max())
            row.update({f"q{int(q*100):02d}": np.quantile(x, q) for q in (.25, .5, .75, .9, .95, .99)})
            # For random, these thresholds apply to the within-query mean degradation.
            row.update({f"query_mean_degradation_ge_{t}": (x >= t).mean() for t in thresholds})
            rows.append(row)
    return pd.DataFrame(rows)


def save_plots(primary, recall, pairs, out):
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#2878b5", "#df6b35", "#42956f"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, ratio, color in zip(axes, RATIOS, colors):
        g = pairs[(pairs.mask_mode == "connected_topk") & (pairs.baseline == "random_token") & (pairs.mask_ratio == ratio)]
        ax.hist(g.paired_margin_difference, bins=35, color=color, alpha=.8)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(g.paired_margin_difference.mean(), color="black", ls="--", label="Mean difference")
        ax.set(title=f"Mask {ratio:.0%} | n={len(g)}", xlabel="Attention − random token margin drop")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Queries")
    fig.suptitle("Paired margin differences: positive = attention more damaging")
    fig.tight_layout(); fig.savefig(out / "paired_margin_distribution.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for j, r in primary.reset_index(drop=True).iterrows():
        ax.errorbar(r.mean_paired_difference, j, xerr=[[r.mean_paired_difference-r.ci_low], [r.ci_high-r.mean_paired_difference]], fmt="o", color=colors[j], capsize=4)
    ax.axvline(0, color="gray", ls="--")
    ax.set(yticks=range(3), yticklabels=[f"{r:.0%}" for r in RATIOS], xlabel="Mean paired margin difference (95% query-bootstrap CI)", title="Connected attention − token random")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out / "margin_effect_forest.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    metrics = ["hit_at_1", "hit_at_5", "hit_at_10", "failure_flip"]
    for ax, subset in zip(axes, ("all_queries", "clean_R1_correct")):
        for j, ratio in enumerate(RATIOS):
            g = recall[(recall.subset == subset) & (recall.mask_ratio == ratio)].set_index("metric").loc[metrics]
            x = g.damage_advantage.to_numpy()*100
            ax.errorbar(x, np.arange(4)+(j-1)*.2, xerr=np.vstack([x-g.ci_low.to_numpy()*100, g.ci_high.to_numpy()*100-x]), fmt="o", capsize=3, color=colors[j], label=f"Mask {ratio:.0%}")
        ax.axvline(0, color="gray", ls="--")
        ax.set(title=f"{subset} (n={int(g.n_queries.iloc[0])})", yticks=range(4), yticklabels=["R@1", "R@5", "R@10", "Failure flip"], xlabel="Damage advantage (percentage points), 95% CI")
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Positive = attention more damaging; random placements averaged within query")
    fig.tight_layout(); fig.savefig(out / "recall_effect_forest.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bottom = np.zeros(3)
    for col, label, color in [("attention_win_rate", "Attention more damaging", "#df6b35"), ("random_win_rate", "Random more damaging", "#2878b5"), ("tie_rate", "Tie", "#aaaaaa")]:
        vals = primary[col].to_numpy()*100
        ax.bar(range(3), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set(xticks=range(3), xticklabels=[f"{r:.0%}" for r in RATIOS], ylim=(0, 110), ylabel="Queries (%)", xlabel="Mask ratio", title="Paired margin win / loss rates")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "win_loss_rate.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, ratio in zip(axes, RATIOS):
        g = pairs[(pairs.mask_mode == "connected_topk") & (pairs.baseline == "random_token") & (pairs.mask_ratio == ratio)]
        for side, label, color in [("attention", "Attention", "#df6b35"), ("random", "Random (query mean)", "#2878b5")]:
            x = np.sort(g[f"{side}_rank_degradation"])
            ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", label=label, color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set(title=f"Mask {ratio:.0%}", xlabel="Rank degradation (symlog)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ECDF")
    fig.tight_layout(); fig.savefig(out / "rank_degradation_ecdf.png", dpi=180); plt.close(fig)


def markdown_table(frame, columns):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.6g}"
        return str(x)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"] +
                     ["| " + " | ".join(fmt(row[c]) for c in columns) + " |" for _, row in frame.iterrows()])


def classify(primary):
    """Descriptive evidence rubric, not a preregistered decision rule."""
    strong = (primary.mean_paired_difference > 0) & (primary.ci_low > 0) & (primary.wilcoxon_p_holm < .05)
    if strong.all():
        return "H1 supported"
    if (primary.mean_paired_difference > 0).any():
        return "H1 weak"
    return "H1 unsupported"


def write_report(primary, secondary, protocol, recall, rank, cohorts, provenance, args, out):
    main = primary[np.isclose(primary.mask_ratio, .15)].iloc[0]
    rec15 = recall[(recall.mask_ratio == .15) & (recall.subset == "all_queries")].set_index("metric")
    verdict = classify(primary)
    lines = ["# Stage 1：现有 ablation500 配对统计", "", f"**{verdict}**（仅限本 checkpoint、数据子集与遮挡 protocol）。", "",
             f"Primary 15%：margin drop 的 attention − random_token 均值差为 **{main.mean_paired_difference:.6f}**，"
             f"95% query-bootstrap CI **[{main.ci_low:.6f}, {main.ci_high:.6f}]**；"
             f"Cohen's dz={main.cohen_dz:.3f}，rank-biserial={main.rank_biserial:.3f}，attention win rate={main.attention_win_rate:.1%}。", "",
             f"15% 的 R@1：attention={rec15.loc['hit_at_1', 'attention_mean']:.2%}，random_token={rec15.loc['hit_at_1', 'random_token_mean']:.2%}。"
             f"failure flip：attention={rec15.loc['failure_flip', 'attention_mean']:.2%}，random_token={rec15.loc['failure_flip', 'random_token_mean']:.2%}。", "",
             "## 数据与统计约定", "",
             "仅读取三份 CSV，没有加载模型、checkpoint、图片或 descriptor cache，没有执行 inference。"
             "逐项从 per_query.csv 重建并核对两个 paired 文件的八类 endpoint；每个合格 query 必须恰有 5 次 random placements。"
             "同一 mask construction 的 5 次 random 值先在 query 内平均；统计样本数为 query 数。", "",
             f"Bootstrap：固定 seed={args.seed}，{args.bootstrap_resamples:,} 次有放回 query 重采样，均值差的 percentile 95% CI。"
             "CI 为逐项区间，不是 simultaneous CI；没有重采样 placement，也不刻画额外 placement 的 Monte Carlo 不确定性。"
             "同一 recall cohort 的四个 endpoint 共用重采样 query 索引。", "",
             "Primary 固定 connected_topk / random_token / 10%、15%、20%，15% 为主要比例。"
             "两种检验均为双侧：Wilcoxon signed-rank 为主检验（asymptotic，zero_method=wilcox，不作连续性校正），"
             "配对 t-test 为均值差的补充检验。分别在三个比例内作 Holm 校正，不把两种检验当作独立复现。"
             "Wilcoxon 的位置差解释依赖差值分布对称性，不作为任意分布下的均值或中位数检验。", "",
             "Cohen's dz = mean(D) / sample_std(D)，rank-biserial = (W+ − W−)/(W+ + W−)，"
             "后者剔除零差、绝对值 ties 用平均秩。仅 rank-test 和 win/loss 将差值四舍五入到小数点后 12 位；"
             "均值、CI、t-test、dz 使用原始差值。win/loss 分母包括 ties。常量非零差值的 dz 未定义，写为空值。", "",
             "这些检验和 bootstrap 将 query 视为独立单位；同路线或近邻 query 的潜在相关性未作 cluster 校正。"
             "因此结果支持当前 intervention experiment 中的 attention proposal，不把 raw attention 称为 causal vulnerability 或 shortcut，"
             "也不外推到其他 checkpoint、数据集或遮挡方式。", "",
             "证据标签采用透明的事后描述规则：三个 primary ratio 均为正且 mean CI 下界 > 0、"
             "Wilcoxon Holm p < 0.05 时为 supported；否则有正向均值时为 weak；其余为 unsupported。"
             "这不是预注册规则，secondary 和 protocol comparison 不用于升级标签。", "",
             "## Primary margin endpoint", "",
             markdown_table(primary, ["mask_ratio", "n_queries", "mean_paired_difference", "median_paired_difference", "ci_low", "ci_high", "cohen_dz", "rank_biserial", "attention_win_rate", "random_win_rate", "tie_rate"]), "",
             markdown_table(primary, ["mask_ratio", "wilcoxon_p_raw", "wilcoxon_p_holm", "t_p_raw", "t_p_holm"]), "",
             "## Recall / failure flip", "",
             "Recall damage advantage = random_token − attention；failure-flip advantage = attention − random_token。"
             "两者正数均代表 attention 更具破坏性。表中率和 CI 用 0–1 单位（乘 100 即百分点）。"
             "clean-R@1-correct 子集由未遮挡 clean rank 固定确定，不按 perturbation 结果筛选。", "",
             markdown_table(recall, ["mask_ratio", "subset", "metric", "n_queries", "attention_mean", "random_token_mean", "damage_advantage", "ci_low", "ci_high"]), "",
             "## Secondary：仅两类比较", "",
             "raw_topk 与自己的 random_token 在可平移 query 上配对；raw_topk 与 connected_topk 直接比较使用全部 query，"
             "包括无法平移的 raw mask。两者 token 与 pixel budget 均已核对一致。Secondary 的六个 contrast "
             "按检验分别额外作 Holm 校正，仍属于 exploratory；完整检验结果见 secondary_statistics.csv。", "",
             markdown_table(secondary, ["comparison", "mask_ratio", "n_queries", "n_excluded", "mean_paired_difference", "ci_low", "ci_high"]), ""]
    for _, r in secondary[secondary.comparison == CONSTRUCTION].iterrows():
        lines.append(f"- raw − connected，{r.mask_ratio:.0%}：**{r.interpretation}**。")
    lines += ["", "CI 跨 0 仅表示 no stable difference detected，不能写成 proved equivalent；未做等效性检验。", "",
              "## Protocol comparison（不属于 primary/secondary H1 证据）", "",
              "以下为 query 内 random_token 均值 − random_pixel 均值的 margin drop。"
              "两者不仅 patch alignment 不同，sampling policy 也不同（token 在位置足够时不放回，pixel 放回），"
              "因此不能解释为纯粹的 patch-alignment causal effect。未对这些探索性 protocol p-values 作校正，也不据此作确认性结论。", "",
              markdown_table(protocol, ["comparison", "mask_ratio", "n_queries", "mean_paired_difference", "ci_low", "ci_high"]), "",
              "## Rank 重尾描述", "",
              "不报告 mean rank degradation 作为 headline。保留 ECDF、median (q50)、分位数和 catastrophic tail。"
              "Random 的 ECDF / 分位数 / tail 均基于每个 query 的 5 次 rank degradation 均值，"
              "tail 表示该 query 均值达到指定阈值的比例，不是单次 placement 的尾概率。", "",
              markdown_table(rank, list(rank.columns)), "",
              "## 配对 cohort 与排除", "", markdown_table(cohorts, list(cohorts.columns)), "",
              "## 可复现命令与输入校验", "", "```bash",
              f"python scripts/analyze_stage1_results.py --input-dir {args.input_dir} --output-dir {args.output_dir} "
              f"--seed {args.seed} --bootstrap-resamples {args.bootstrap_resamples} --expected-queries {args.expected_queries} "
              f"--tail-thresholds {' '.join(map(str, args.tail_thresholds))}", "```", "",
              "输入 SHA-256 和软件版本见 analysis_manifest.json；运行结束再次验证三份输入未改变。", ""]
    lines += [f"- `{name}`：`{value}`" for name, value in provenance["input_sha256"].items()]
    lines += ["", "## 图表", ""]
    for name in ("paired_margin_distribution", "margin_effect_forest", "recall_effect_forest", "win_loss_rate", "rank_degradation_ecdf"):
        lines += [f"![{name}](plots/{name}.png)", ""]
    (out / "STATISTICAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/stage1/ablation500"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1/statistics500"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--expected-queries", type=int, default=500)
    parser.add_argument("--tail-thresholds", type=int, nargs="+", default=[100, 1000])
    args = parser.parse_args(argv)
    require(args.bootstrap_resamples >= 10000, "Formal statistics require >= 10,000 query resamples")
    require(args.seed >= 0 and args.expected_queries >= 2, "Invalid seed or cohort size")
    require(all(t > 0 for t in args.tail_thresholds), "Tail thresholds must be positive")
    require(args.input_dir.resolve() != args.output_dir.resolve(), "Keep input and output directories separate")
    paths = [args.input_dir / f for f in ("per_query.csv", "paired_query.csv", "ablation_paired_query.csv")]
    hashes = {p.name: file_hash(p) for p in paths}
    per, pairs, ablations = [pd.read_csv(p, low_memory=False) for p in paths]
    clean, cohorts = validate_inputs(per, pairs, ablations, args.expected_queries)
    print(f"Validated {len(clean)} queries, all five-placement means and both paired files.", flush=True)
    primary, secondary, protocol = margin_tables(pairs, ablations, args, len(clean))
    recall = recall_table(pairs, args)
    rank = rank_table(pairs, args.tail_thresholds)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("primary_statistics", primary), ("secondary_statistics", secondary),
                        ("recall_bootstrap", recall), ("protocol_comparison", protocol),
                        ("rank_statistics", rank), ("validated_cohorts", cohorts)]:
        frame.to_csv(out / f"{name}.csv", index=False)
    save_plots(primary, recall, pairs, out / "plots")
    require(hashes == {p.name: file_hash(p) for p in paths}, "Input CSVs changed during analysis")
    manifest = dict(input_dir=str(args.input_dir.resolve()), input_sha256=hashes,
                    input_rows={p.name: len(f) for p, f in zip(paths, [per, pairs, ablations])},
                    bootstrap_resamples=args.bootstrap_resamples, seed=args.seed,
                    bootstrap_unit="query", random_repeats=5, primary_ratios=RATIOS,
                    primary_mask_mode="connected_topk", primary_baseline="random_token",
                    tail_thresholds=args.tail_thresholds, no_model_inference=True,
                    source_sha256=file_hash(Path(__file__)),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    verdict = write_report(primary, secondary, protocol, recall, rank, cohorts, manifest, args, out)
    manifest["verdict"] = verdict
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(primary[["mask_ratio", "mean_paired_difference", "ci_low", "ci_high", "wilcoxon_p_holm"]].to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
