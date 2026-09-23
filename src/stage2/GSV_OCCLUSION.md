# GSV targeted vs shape-matched random gate

新增脚本 `scripts/stage2_eval_gsv_occlusion.py` 和实验工具
`src/stage2/gsv_occlusion.py`。复用 Prompt 3 的估计器，以及既有 BoQ forward、
`RetrievalContext.query`、RGB mean-fill 和 rigid translation；没有改动模型或检索公式。
随机平移函数实际位于 `src/analysis/perturb.py`。

## 运行

```bash
# 主检验：沿用 Prompt 2 推荐的 SUPPORT 图片库。
conda run --no-capture-output -n boq python scripts/stage2_eval_gsv_occlusion.py

# 敏感性分析：相同图像、固定 seed，切换为支持集均值归一化原型。
conda run --no-capture-output -n boq python scripts/stage2_eval_gsv_occlusion.py \
  --reference-mode prototype --output-dir outputs/stage2/gsv_occlusion/prototype

conda run -n boq pytest tests/test_stage2_gsv_occlusion.py
```

默认 `--seeds 0 1 --num-queries 50 --random-repeats 5 --bootstrap-resamples 20000`。
`--mask-ratio .15 --top-k 32 --alpha .5` 与 Prompt 3 一致。
模型、RGB 预处理、split、图片文件信息、缓存身份均重新核验；默认引用完整
SUPPORT 库，也支持 `--reference-mode prototype`。
`--retrieval-report`、`--split`、`--output-dir` 可指向其他位置。

## 固定实验设计

1. 对每个 seed，先对全局 place key 做 SHA256 确定性随机排序，再在每个选中的
   place 中选择一个按 SHA256 排序的 SOURCE。每组 50 个不同 place，避免同一
   place 的多个视图重复充当独立 bootstrap 单位。
2. Attention 是主检验，Fused 是补充检验。只运行一次 Prompt 3 探测，然后用
   已有 connected-topk 从 attention map 和 fused map 构造两张 mask。
3. 每张 mask 的随机对照都是**它自身的精确像素平移**，不裁切、不旋转、不膨胀。
   在全部合法整数像素位移中有放回均匀采样 5 次，排除原位置；允许与原位置重叠。
   不同策略使用各自形状匹配的对照。随机位移不要求 token 对齐，token 数只描述
   target mask；像素面积和完整轮廓严格一致。
4. 原始 RGB 通道均值作为所有条件的共同填充值，之后执行 ImageNet Normalize。
   Clean、两种 Targeted 及各自 5 个 Random 总共 13 张图分批编码，使用相同
   全局检索库。clean 描述符与估计器基线之差超过 `2e-6` 时直接停止。
5. 保留有符号 margin drop、描述符 drift、正例相似度变化、rank degradation、
   Recall 命中与 failure flip。负损害不能裁成 0。

仅在目标 mask 没有任何合法非零平移时，该 query 才退出相应策略的配对统计。
保留原始 clean/targeted 行和剔除原因，不改形状“补足”随机样本。
估计器使用 `strict=False` 保存诊断；不会因为窗口损害较小、为零或 STOP 而排除
query，以免根据实验结果筛选样本。非有限描述符、预算/形状错误仍直接失败。

## 统计和 Stop/Go

每个 SOURCE 的 5 个随机结果先求均值，再形成
`D = targeted_margin_drop - mean(random_margin_drop)`。D>0 表示目标化遮挡更有害。
复用 Stage1 的 `paired_statistics`：20,000 次 query-paired bootstrap、均值差的
percentile 95% CI、双尾 Wilcoxon。Wilcoxon 沿用 Stage1 的 asymptotic 方法，
去除零差、对 rank-test 输入舍入到 12 位；不修改 bootstrap 的原始差值。

Attention 和 Fused 各自形成跨 seed 的 Holm 校正族。主关卡要求**每个预先指定的
seed** 都满足 Attention：均值差>0、95% CI 下界>0、Holm p<0.05、配对覆盖率≥80%。
至少两个 seed 才能通过一致性关卡；单 seed 运行会保存结果但标记 STOP。
CI 包含 0、显著性不足或配对不足时，退出码非零，不能继续生成阶段。

Fused 的统计结果单独报告；`fused_estimator_ready` 还要求所有估计器诊断通过。
SUPPORT 是主数据库，原型库是预先声明的敏感性分析，不用其中较有利的结果替代
主检验。改变 seed 同时改变 SOURCE 抽样和随机位移；本次没有重新分配 SUPPORT。
其他 place 间的地理相关性未作空间 cluster 校正，结论限定于当前 GSV 协议。

## 输出

每个 reference mode 一个输出目录：

- `config.json`：全部参数、固定 source cohorts、缓存/模型/数据/代码身份和判据。
- `per_query.csv`：每个 clean、targeted、random draw 的原始指标、面积、位移与资格。
- `paired_queries.csv`：图内平均后的随机指标及配对差，一行一个 query/strategy/seed。
- `group_summary.csv`：同一有效配对 cohort 上每种条件的 query-level 均值、样本方差。
- `paired_statistics.csv`、`summary.json`：CI、双尾 p、Holm p、胜率和 Stop/Go。
- `exclusions.json`、`estimator_diagnostics.json`：所有剔除和弱信号诊断，不能静默过滤。
- `target_masks_seed*.npz`：目标 token masks；配合 CSV 的 seed/位移可重建随机对照。
- `diagnostics_seed*.png`：散点、CDF、Clean/Random/Targeted 箱线图。

验证结果见 [RESULTS.md](../../outputs/stage2/gsv_occlusion/RESULTS.md)。
