# GSV vulnerability validation — GO

**当前 GSV 协议支持继续下一阶段的生成级编辑验证。** 固定 checkpoint、精确同形状
同面积随机对照下，Attention-targeted 在两组独立抽样的 SOURCE 上均造成更大的
margin drop；两组 95% CI 均完全高于 0，Holm 校正后的双尾 Wilcoxon 均显著。
原型库敏感性分析保持相同结论。Fused 补充检验与估计器诊断也全部通过。

## 固定协议与主检验

沿用 Stage1 冻结 DINOv2 ViT-B/14 + BoQ checkpoint、224×224、float32、ImageNet
归一化。主检索库为 127,118 张 SUPPORT，敏感性分析为 63,559 个 place 原型。
split seed=0 固定，SOURCE 不参与任何 reference 或原型。

预先指定实验 seed=0、1，各抽 50 个不同 place 的 SOURCE。两组之间 **0 个 place
重叠**。每张图的每种策略使用 5 次均匀随机平移，先在图内求均值，再进行配对统计。
目标预算为 38 tokens、7,448 pixels（14.84375%）；所有随机对照保持其自身目标的
精确像素轮廓和面积，不保证 token 对齐。无任何样本因几何不可平移而剔除。

主统计量：`Targeted margin drop − mean(Random margin drop)`，正值表示目标化更有害。
Bootstrap：20,000 次 query 重采样；Wilcoxon：双尾、asymptotic，按策略跨 seed 做 Holm 校正。

| SUPPORT 库 / Attention | 有效配对 | 平均损害优势 | 95% paired bootstrap CI | 双尾 p | Holm p |
|---|---:|---:|---:|---:|---:|
| seed=0 | 50/50 | 0.046143 | [0.033258, 0.058998] | 1.63×10⁻⁷ | 3.27×10⁻⁷ |
| seed=1 | 50/50 | 0.028094 | [0.016332, 0.040238] | 1.11×10⁻⁴ | 1.11×10⁻⁴ |

逐图胜率分别为 **84%**、**74%**。两组均满足预设 GO 条件：均值差>0、CI 下界>0、
Holm p<0.05、有效配对比例≥80%。

## 原型库一致性与 Fused 补充检验

原型库使用相同 SOURCE、Attention masks 和随机位移；只改变 reference 表示。
这是引用库表示的敏感性分析，本次没有改变 SUPPORT 图片分配。

| 引用库 | 策略 | seed | 平均损害优势 | 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| Prototype | Attention | 0 | 0.043338 | [0.030709, 0.055832] | 8.68×10⁻⁷ |
| Prototype | Attention | 1 | 0.027660 | [0.015746, 0.040207] | 1.20×10⁻⁴ |
| SUPPORT | Fused | 0 | 0.056559 | [0.043248, 0.069895] | 3.36×10⁻⁸ |
| SUPPORT | Fused | 1 | 0.046295 | [0.031980, 0.061063] | 1.42×10⁻⁶ |
| Prototype | Fused | 0 | 0.051346 | [0.038650, 0.063730] | 7.29×10⁻⁸ |
| Prototype | Fused | 1 | 0.045238 | [0.032493, 0.058328] | 2.89×10⁻⁷ |

上述 Fused 差异均相对于 **Fused 自身形状的随机对照**。它与 Attention 使用的
形状不同，这张表不是 Fused 相对 Attention 的直接优越性检验。

## 工程和诊断核验

- 每种引用库保存 1,400 行原始条件/重复指标、200 行配对结果；两种库共 2,800 行。
- 主检验和敏感性分析均无剔除、无 Prompt 3 STOP。没有根据窗口损害强弱筛选图像。
- 最大 clean 描述符批量差 `2.42×10⁻⁷`，低于 `2×10⁻⁶` 容差。
- 独立从 CSV 重建 bootstrap/Wilcoxon 结果，并重新生成全部 **2,000 张随机 mask**，
  核对 seed、偏移、完整轮廓、面积、margin 记账和代码哈希，全部通过。
- 新增 8 个测试通过，全仓库 **76 个测试通过**。覆盖配对单位、两侧检验、零效应/反向
  效应 STOP、缺失/重复 random draw 拒绝、样本一致性、同形随机、不可平移统计与弱信号保留。

散点、CDF 和箱线图显示 Targeted 的整体损害较高，但分布仍有重叠，部分 query
存在零或负损害；这些结果全部保留。GO 依据配对 CI 和检验，不要求每张图都胜过随机。

## 产物和复现

- [SUPPORT 配置](support/config.json)、[统计结论](support/summary.json)、[原始指标](support/per_query.csv)、
  [均值/方差](support/group_summary.csv)、[配对结果](support/paired_queries.csv)。
- [seed=0 诊断图](support/diagnostics_seed0.png)、[seed=1 诊断图](support/diagnostics_seed1.png)。
- [原型库统计](prototype/summary.json)、[原型库 seed=0 图](prototype/diagnostics_seed0.png)、
  [原型库 seed=1 图](prototype/diagnostics_seed1.png)。
- [独立核验](verification.json)、[运行与 API 说明](../../../src/stage2/GSV_OCCLUSION.md)。

```bash
conda run --no-capture-output -n boq python scripts/stage2_eval_gsv_occlusion.py
conda run --no-capture-output -n boq python scripts/stage2_eval_gsv_occlusion.py \
  --reference-mode prototype --output-dir outputs/stage2/gsv_occlusion/prototype
```

结论限定于当前 checkpoint、GSV 域、15% connected mask、mean-fill 和像素平移
对照。每个 place 只抽一张图；邻近 place 的空间相关性未进一步 cluster 校正。
本关卡证明目标化遮挡有害，尚不能证明生成式编辑能够保持地点身份或提升训练效果。
