# 三类案例的人工查看记录

查看了 query 250 的总览，并对以下三张已保存案例图逐一检查 RGB、raw attention、signed deletion map、overlay 和 scatter。另外从 per_window.csv 独立核对了全部 9 个保存案例的分组条件。这里的 strong / weak 是同一 query 内的相对分位，不能理解为跨图像的绝对强度，也不一定是图中最亮的位置。

| 类别 | Query / window (y,x) | Attention percentile | Damage percentile | Signed margin drop |
| --- | --- | --- | --- | --- |
| attention strong + intervention strong | 350 / (0,21) | 82.61 | 88.20 | +0.006124 |
| attention strong + intervention weak | 505 / (16,18) | 89.23 | 11.18 | −0.000545 |
| attention weak + intervention strong | 258 / (18,17) | 0.21 | 83.64 | +0.003250 |

百分位均以 ascending average rank 计算，100 表示该 query 内最高；绿色框与 scatter 的绿色星号是同一 window。三例的 best-positive rank 都保持 1，说明连续 margin 指标能记录未触发检索失败的变化。

## Attention strong + intervention strong

[Query 350 图](visualizations/cases/attention_strong_intervention_strong_query350.png)中，选中窗口位于右上方建筑边缘。它不在最亮的中心 attention 峰上，但 window attention 位于该 query 的上 20%，deletion damage 也位于上 20% 且为正。中心附近另有更大的正、负 damage，不能从单个一致案例推断整张 attention 图准确。

## Attention strong + intervention weak

[Query 505 图](visualizations/cases/attention_strong_intervention_weak_query505.png)中，选中窗口在右侧路边、骑行者旁。Attention 位于第 89.23 百分位，而 margin drop 为负，即此次遮挡使 margin 略有增加。图中该 query 整体 Spearman 约 0.484，仍出现这个局部反例；正相关不等于每个高响应位置都敏感。

## Attention weak + intervention strong

[Query 258 图](visualizations/cases/attention_weak_intervention_strong_query258.png)中，选中窗口位于右下方路面。Attention 很低，但 deletion damage 为正且位于第 83.64 百分位；该 query 整体 Spearman 约 −0.045。这里只记录窗口遮挡与 retrieval margin 的关系，不把路面等语义对象解释为独立因果因素。

## 可视化检查

- Deletion map 的色标以 0 为中心，负值保留为蓝色，正值为红色；scatter 也保留全部负 damage。
- Window-center 图显示真实窗口结果；overlay 对重叠窗口作均值，仅供显示，没有用于 Spearman 或 top/bottom 统计。
- 不同 query 的 colorbar 范围不同，应按标注数值比较，不能直接比较颜色饱和度。
- 三类各保存 3 例，均来自固定 seed 的 query reservoir sampling；以上仅为其中各一例的查看记录，不替代 100-query 汇总。

本次结果存在稳定的正向 proposal 信息，但平均 Spearman 只有 0.162，局部反例明显。因此采用较保守结论：**attention is useful for targeted occlusion globally, but is not sufficiently faithful as a local vulnerability estimator**。不生成 causal map，不做 fusion。
