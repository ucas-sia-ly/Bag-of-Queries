# GSV retrieval context — GO

使用 Prompt 1 的 seed=0 清单和 Stage1 冻结 checkpoint：
`logs/dinov2_vitb14/version_2/checkpoints/epoch[21]_R@1[0.9068]_R@5[0.9459].ckpt`。
模型为 DINOv2 ViT-B/14 + BoQ，描述符维度 8192；224×224 bicubic antialias、
ImageNet 归一化、float32、eval、无随机增强、TF32/xformers 关闭。
224 是该 checkpoint 的训练尺寸；训练器对照使用相同预处理和描述符。

完整 SUPPORT 缓存：[`.cache/stage2/gsv_support.pt`](../../../.cache/stage2/gsv_support.pt)。
它保存在本地，不纳入 Git；复现命令见 [使用说明](../../../src/stage2/README.md)。
所有 127,118 张 SUPPORT 均成功解码，缓存形状 `[127118, 8192]`，
所有描述符有限且 L2 norm≈1。缓存重载和输入身份、行序、维度校验通过。

## 100 张 SOURCE 全库验证

SOURCE 使用 seed=0 的 SHA256 顺序抽样；两种模式使用完全相同的 100 张图。
正例属于同一 place，负例覆盖其余所有城市/place。没有 SOURCE 进入引用库。

| 指标 | SUPPORT 图片库 | Place 原型库 |
|---|---:|---:|
| Reference 数 | 127,118 | 63,559 |
| R@1 | 97% | 99% |
| R@5 / R@10 | 100% / 100% | 100% / 100% |
| 平均 margin | 0.230527 | 0.248310 |
| 与原训练器 FAISS R@1/5/10 | 完全一致 | 完全一致 |
| 独立余弦核验最大绝对误差 | 6.86×10⁻⁸ | 5.65×10⁻⁸ |

训练器原始函数没有 margin；margin、正/负相似度直接复用 Stage1 接口。
对前 3 个 SOURCE，额外用 NumPy float64 全库余弦独立验证这些数值，均小于
预设容差 `2e-5`。逐图结果见 [support_queries.csv](support_queries.csv) 和
[prototype_queries.csv](prototype_queries.csv)，完整身份、SHA256、查询列表见
[validation.json](validation.json)。

## 原型检查与模式选择

固定 seed 随机抽取 5 个 place，每个均有 2 张 SUPPORT：

| Place | normalize(mean(SUPPORT)) 的 L2 norm |
|---|---:|
| Minneapolis:5259 | 0.999999821 |
| Miami:1126 | 1.000000119 |
| Lisbon:4833 | 1.000000000 |
| Lisbon:8723 | 1.000000000 |
| Phoenix:1960 | 1.000000000 |

两种库的 margin Spearman 为 **0.930957**，符号一致率 **98%**，
margin 平均绝对差 **0.036611**，最大 Recall 差 **2 个百分点**。
没有同时满足事先设定的 Spearman≥0.95、符号一致率≥0.95、Recall 差≤1 个百分点。
因此后续默认使用 **SUPPORT 图片库**；原型接口保留为可选方案。

## 工程验证

- 13 个新增检索测试通过，全仓库 58 个测试通过。
- dummy BoQ 固定特征可复现手算排序、margin 和原训练器 Recall。
- 预处理与原始 VPRDataModule validation transform 逐像素相同。
- 覆盖缓存身份/行序失配、SOURCE 改动、角色泄漏、坏图、非有限/零向量和非法原型。
- 实际 4.2 GB 缓存在中文路径完成写入与重载；采用缓冲写入，并在完整校验后原子发布。
- 热加载验证见 [cache_reuse_check.json](cache_reuse_check.json)：不重新编码 SUPPORT，
  缓存文件 SHA256 保持一致。

这些 GSV 结果是检索管线的一致性验收，不是独立测试集的泛化成绩。
