# Stage 2 Source/Support split

在仓库根目录运行（构建脚本仅依赖 Python 标准库）：

```bash
python scripts/stage2_build_split.py
conda run -n boq pytest tests/test_stage2_split.py
```

默认读取 `data/train/gsv-cities/Dataframes/*.csv`，seed 为 0，输出到
`outputs/stage2/split/gsv_split.jsonl` 和 `gsv_split_summary.json`。
支持其他数据位置、城市子集和独立输出目录：

```bash
python scripts/stage2_build_split.py \
  --dataframes-dir /path/to/gsv-cities/Dataframes \
  --cities Bangkok London --seed 42 --output-dir /tmp/gsv-split-42
```

API：

```python
from src.stage2.data.gsv_split import build_split

manifest = build_split(["Bangkok", "London"], seed=0)
# 可选关键字参数 dataframes_dir 指向另一个 Dataframes 目录。
manifest.write("outputs/stage2/split")
```

## 划分协议

- `place_key = f"{city}:{local_place_id}"`，城市之间的本地 ID 不混合。
- `image_key` 是相对于 GSV-Cities `Images/` 的图片路径，使用已有文件命名格式，
  包含城市、place、year、month、northdeg、lat、lon、panoid。
  经纬度保留 CSV 文本，northdeg 保留负数和大于 360 的原值。
- 完全相同的图片键先去重；独立图片不足 3 张的 place 整体剔除。
- 对每个 place，按 `SHA256(UTF8(str(seed) + "\0" + image_key))` 升序排序，
  哈希相同时用 image_key 排序。前 2 张为 `SUPPORT`，其余为 `SOURCE`。
- 输出顺序为城市名、本地 place ID、上述哈希顺序。清单行包含
  `image_key, place_key, city_id, local_place_id, role`；`city_id` 是城市名，
  `local_place_id` 是整数。
- SOURCE 与 SUPPORT 在图片级别不重叠，同一 place 的所有 SOURCE 共享 2 张 SUPPORT。
  这是 place 内的角色划分，不是 place 互斥的训练/测试划分。

同一 seed 的清单字节可复现，不受 CSV 行序、城市参数顺序、额外城市或
`PYTHONHASHSEED` 影响。不同 seed 改变哈希排序，小样本上的角色分配仍可能碰巧一致。
summary 记录 seed、协议、城市、总计及分城市统计、剔除 ID、视图数分布、原始 CSV
SHA256 和清单文件 SHA256。CSV 行序变化会改变输入文件 SHA256，但不改变清单。

## Stop/Go 和本次验证

固定 `K_support=2`。如果某城市为空、没有有效 place，或超过 50% 的 place
不足 3 张独立图片，脚本以非零状态退出，且不写输出。缺失 CSV、列或图片标识字段
同样报错。停止后应审查城市或实验协议，不自动降低 K_support。

全量 seed=0 的结果：23 个城市，63,559 个有效 place，835 个剔除 place，
527,836 行，其中 SOURCE 400,718 张、SUPPORT 127,118 张，重复行 0。
最高剔除比例为 PRG 的 10.854%，检查结果为 GO。

独立读取生成清单验证了图片键唯一、角色互斥、每个 SOURCE place 恰好 2 张 SUPPORT、
统计及 SHA256 一致，并核对所有 527,836 个图片路径均存在。
构建器本身只读 CSV，不要求图片下载完毕；上述图片存在性检查针对当前本地数据。

SUPPORT 总数可以小于 SOURCE 总数，因为同一 place 的 SOURCE 共享 SUPPORT。
验收条件是每个 SOURCE 对应 2 张 SUPPORT，而非两类图片总数的大小关系。
