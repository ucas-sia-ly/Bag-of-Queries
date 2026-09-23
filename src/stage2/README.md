# Stage 2 GSV retrieval context

后续 attention/intervention 脆弱性估计器见 [VULNERABILITY.md](VULNERABILITY.md)。

读取 [Source/Support 清单](data/README.md)，只编码 SUPPORT，并复用
`src/analysis/retrieval.py::retrieval_metrics` 计算全库检索指标。
SOURCE 从不参与 reference 或 place 原型。每个 query 的正例是同一 `city:place_id`
的全部 references，负例是所有其他 place 的 references，不做城市内截断或负例采样。

## 构建与验证

```bash
conda run --no-capture-output -n boq python scripts/stage2_build_retrieval.py \
  --checkpoint 'logs/dinov2_vitb14/version_2/checkpoints/epoch[21]_R@1[0.9068]_R@5[0.9459].ckpt'

conda run -n boq pytest tests/test_stage2_split.py tests/test_stage2_retrieval.py
```

默认读取 `outputs/stage2/split/gsv_split.jsonl`，图像根目录为
`data/train/gsv-cities/Images`，输出：

- `.cache/stage2/gsv_support.pt`：有序 SUPPORT 描述符、image/place keys、完整缓存身份。
- `outputs/stage2/retrieval/validation.json`：输入身份、缓存 SHA256、100 个 SOURCE
  的抽样清单、Stop/Go、原始训练器 Recall 对照、余弦 oracle、5 个原型 norm 和模式比较。
- 同目录的 `support_queries.csv`、`prototype_queries.csv`：每张 SOURCE 的 rank、
  正/负相似度、margin、R@1/5/10 命中标志。

`--split`、`--images-root`、`--cache`、`--output-dir` 可指向其他 workspace。
`--image-size H W`、`--batch-size`、`--workers`、`--device` 可配置；DINO 输入边长必须为
patch size 的倍数。`--num-queries` 默认 100，`--seed` 默认 0，SOURCE 按
`SHA256(seed + NUL + image_key)` 排序取前 N 个。
`--cache-only` 仅构建缓存，不代表完整检索验收通过。

模型严格复用 Stage1 的 `load_model`，由 checkpoint 推断 BoQ 维度并 strict load，
冻结参数、eval、float32、关闭 TF32/xformers、启用确定性计算。
默认 224×224 来自当前 checkpoint 的训练尺寸；预处理是原训练器 validation 的
RGB uint8 → bicubic antialias resize → float32 /255 → ImageNet Normalize。
不使用训练时的随机 RandAugment。若要采用 Stage1 的 322×322 评估尺寸，
需显式 `--image-size 322 322` 并使用新的缓存路径。

每批输出必须为有限、非零、L2 norm≈1 的 float32 `[N,D]`。编码不静默跳过坏图。
缓存以磁盘映射构建，通过带缓冲的文件流保存，完整重载校验后原子替换；
这也避免当前 PyTorch 在中文路径下对 >2 GiB 原始写入发生截断。
加载使用 `torch.load(weights_only=True, mmap=True)`。
它校验 split、checkpoint、图像顺序和 place 映射、所有 SUPPORT 的文件 size/mtime、
预处理、模型/编码源代码、Torch/Torchvision、设备、batch size、形状和向量数值。
身份不一致直接报错，使用新的 `--cache` 路径重新构建。
图片文件指纹基于路径/size/mtime，不是每张图片的内容 SHA256。

## 查询 API

```python
import json
from pathlib import Path
from src.stage2.retrieval import GSVSplit, load_support_cache, RetrievalContext

split = GSVSplit.read("outputs/stage2/split/gsv_split.jsonl")
report = json.loads(Path("outputs/stage2/retrieval/validation.json").read_text())
assert report["status"] == "GO"
descriptors = load_support_cache(report["cache_path"], split, report["identity"])
context = RetrievalContext.from_support(descriptors, split.support, mode="support", device="cuda")

# query_descriptor 由报告中相同的冻结 checkpoint 和预处理生成，shape=[D]，L2 归一化。
metrics = context.query(query_descriptor, source_record.place_key)[0]
positives = context.positive_indices(source_record.place_key)
negatives = context.negative_indices(source_record.place_key)

# 同一 SOURCE 的多个编辑版本：[num_variants,D]，显式提供原始描述符用于 drift。
variants_metrics = context.query(variant_descriptors, source_record.place_key,
                                 clean_descriptor=query_descriptor)
```

上述加载方式消费已经验证的固定 artifact；如果本地图像或模型可能已变更，先重新运行
构建脚本，或调用 `cache_identity(...)` 重算预期身份，再传给 `load_support_cache`。
不能把两个 checkpoint 或预处理不同的 query/reference 描述符混用。

`mode="prototype"` 会按 place 汇总 SUPPORT：
`normalize(mean(d_1, d_2, ...))`，每个 place 一条 reference。原型 place keys
排序稳定，零均值无法归一化时停止。可直接调用 `place_prototypes` 取得 place 列表和
原型矩阵；它们按需从 SUPPORT 缓存生成，不额外保存冗余的原型缓存。

## 验收含义

训练器的 `src/utils.py::compute_recall_performance` 只计算 FAISS L2 Recall，没有
margin。因此对相同 query/reference 描述符和 ground truth：

1. R@1/5/10 必须与原始训练器函数完全一致。
2. margin 等公式复用 Stage1，不在生产 API 中另写一套。验证代码独立用 NumPy
   float64 全库余弦核验前 3 个真实 query 的正/负相似度与 margin，容差 `2e-5`。
3. 单元测试用固定输出的 dummy BoQ 重现手算排序和 margin，验证 Source/Support
   隔离、全局正/负索引、预处理一致性、缓存失配拒绝和损坏图像失败行为。
4. 从所有原型中固定 seed 随机抽 5 个 place，报告其 SUPPORT 数和原型 norm。

任一数值/Recall 校验失败都停止，不标记 GO。Stage1 精确相似度相同时按 reference
index 排序；FAISS 对完全并列的索引排序不保证一致，若真实验证命中并列导致 Recall
不一致，仍会触发 STOP 供检查。

同时比较 SUPPORT 库与原型库的 margin Spearman、margin 正负号一致率、Recall 差异。
事先设定推荐原型库的条件为 Spearman≥0.95、符号一致率≥0.95、最大 Recall 差≤0.01；
达不到就保留 SUPPORT 库。两个模式本身都要通过训练器/余弦一致性检查。
这些 GSV 结果用于验证检索管线，不是独立测试集的泛化成绩。
