# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.datamodule import VPRDataModule

class HyperParams:
    def __init__(self):
        ## Backbone config:
        self.backbone_name: str = "dinov2_vitb14"    # resnet18, resnet50, dinov2_vits14, dinov2_vitl14
        self.unfreeze_n_blocks: int = 2              # number of blocks to unfreeze in the backbone

        ## BoQ config:
        self.channel_proj: int = 512
        self.num_queries: int = 64
        self.num_layers: int = 2
        self.output_dim: int = 8192

        ## Datasets:
        # NOTE: if you already have OpenVPRLab, you can set the path to the datasets from there
        # otherwise use the dowload scripts in `scripts/` to download to `data/` folder
        self.gsv_cities_path: str = "data/train/gsv-cities"    # path to gsv-cities in OpenVPRLab
        # gsv_cities_path: str = "./data/train/gsv-cities"                   # or path to gsv-cities in this project

        # self.cities: str | list = "all" # train on all cities
        self.cities: str | list = ["Bangkok", "Boston", "PRS"] # train on a subset of cities (check the gsv-cities folder)

        self.val_sets: dict = {
            "msls-val":     "data/val/msls-val",              # path to the msls-val dataset
            "pitts30k-val": "data/val/pitts30k-val",          # path to the pitts30k-val dataset
        }

        ## Training config:
        self.batch_size: int = 128           # batch size is the number of places per batch
        self.img_per_place: int = 4          # number of images per place
        self.max_epochs: int = 40
        self.warmup_epochs: int = 10         # number of linear warmup epochs (not iterations)
        self.lr: float = 1e-4                # learning rate
        self.weight_decay: float = 1e-4
        self.lr_mul: float = 0.1
        self.milestones: list = [10, 20]
        self.num_workers: int = 8

        ## misc
        self.silent: bool = False            # disable console output
        self.compile: bool = False           # compile the model using torch.compile() [experimental]
        self.seed: int = 2024                # random seed for reproducibility

def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)

    # Instantiate the backbone and define the image size for training and validation
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name # in case the user passed dinov2 without the version
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size

    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, crop_last_block=True)
        train_img_size = (320, 320)
        val_img_size = (384, 384)
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size

    else:
        raise ValueError(f"backbone {hparams.backbone_name} not recognized or not implemented!")


    # Instantiate BoQ aggregator
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim//hparams.channel_proj,
    )

    # Define the entire Lightning model for training and validation
    model = BoQModel(
        backbone,
        aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )

    if hparams.compile:
        model = torch.compile(model)



    # Define the datamodule for handling training and validation datasets
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )

    # If you want to display the datasets and training configs
    if not hparams.silent:
        datamodule.setup()                  # first init the datasets
        display_datasets_stats(datamodule)  # then display the stats

    # we use Tensorboard for logging (integrated with PyTorch Lightning)
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False
    )

    # let's save all the hyperparameters to the the log file
    # this will be saved in the logs folder
    # e.g. ./logs/dinov2_vitb14/version_0/hparams.yaml
    tensorboard_logger.log_hyperparams(hparams.__dict__)

    # Define the checkpointing callback
    checkpointing = callbacks.ModelCheckpoint(
        monitor="msls-val/R@1",  # <==== monitor the Recall@1 on the msls-val dataset
        filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )

    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()

    # Lightning Trainer will take a list of callbacks
    callback_list = [checkpointing]
    if not hparams.silent:
        callback_list.append(program_bar)

    # Define the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        precision="16-mixed",
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    # =========================
    # 通用运行参数
    # =========================

    # 开启快速开发模式（Fast Dev Run）。
    # 启用后通常只会执行 1 个训练 iteration 和 1 个验证 iteration，
    # 主要用于快速检查：
    #   1. 数据是否能正常加载
    #   2. 模型前向传播是否正常
    #   3. loss 是否能够正常计算
    #   4. validation 流程是否有报错
    # 不适合用于正式训练。
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable fast dev run (one train and validation iteration)."
    )

    # 静默模式。
    # 启用后减少或关闭终端中的日志输出，
    # 适合不希望控制台打印大量训练信息时使用。
    #
    # 使用方式：
    #   python train.py --silent
    #
    # store_true 表示：
    #   没有传入 --silent -> False
    #   传入 --silent     -> True
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Disable console output."
    )

    # 是否使用 PyTorch 2.x 的 torch.compile() 对模型进行编译。
    #
    # torch.compile() 会尝试对计算图进行优化，
    # 在部分 GPU / PyTorch 版本 / 模型结构下可以提高训练或推理速度。
    #
    # 但需要注意：
    #   1. 第一次运行会有额外的编译开销
    #   2. 某些算子可能不支持 compile
    #   3. 调试时可能使报错信息变得更复杂
    #   4. 有时会额外占用一定 CPU RAM / GPU 显存
    #
    # 因此复现实验初期建议先不开启，确认训练正常后再测试。
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model using torch.compile()"
    )

    # =========================
    # 随机性 / 可复现性参数
    # =========================

    # 随机种子（Random Seed）。
    #
    # 用于控制训练过程中的随机因素，例如：
    #   - 权重初始化
    #   - 数据 shuffle
    #   - 随机数据增强
    #   - batch 采样
    #
    # 设置固定 seed 可以提高实验的可复现性。
    #
    # 例如：
    #   python train.py --seed 42
    #
    # 注意：
    # 固定随机种子并不一定能够保证在不同 GPU、CUDA、
    # cuDNN 或 PyTorch 版本之间得到完全一致的结果。
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility."
    )

    # =========================
    # 训练超参数
    # =========================

    # Batch Size。
    #
    # 表示每个训练 iteration 中使用的 batch 大小。
    #
    # 对 BoQ / GSV-Cities 这类 place recognition 训练尤其需要注意：
    # 某些数据集的一个 "batch item" 可能对应一个 place，
    # 而每个 place 内部又包含多张图片。
    #
    # 例如：
    #   bs = 32
    #   每个 place 采样 4 张图
    #
    # 那么一次 iteration 实际可能加载：
    #   32 × 4 = 128 张图片
    #
    # Batch Size 越大：
    #   + 梯度估计通常更稳定
    #   + GPU 利用率可能更高
    #   - GPU 显存占用增加
    #   - CPU RAM 占用也可能明显增加
    #
    # 如果遇到 CUDA OOM 或系统 RAM 爆掉，
    # 这是首先应该降低的参数之一。
    parser.add_argument(
        "--bs",
        type=int,
        help="Batch size."
    )

    # Learning Rate，学习率。
    #
    # 决定优化器每次参数更新的步长。
    #
    # 学习率过大：
    #   - loss 可能震荡
    #   - 甚至无法收敛
    #
    # 学习率过小：
    #   - 收敛速度很慢
    #   - 可能需要更多 epoch
    #
    # 示例：
    #   --lr 0.0001
    #   --lr 1e-4
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning Rate."
    )

    # Weight Decay，权重衰减。
    #
    # 通常用于 AdamW 等优化器中的正则化，
    # 用于限制模型参数过度增长，从而减轻过拟合。
    #
    # 常见取值例如：
    #   1e-4
    #   1e-3
    #   1e-2
    #
    # 具体数值应尽量按照论文 / 官方配置进行复现。
    parser.add_argument(
        "--wd",
        type=float,
        help="Weight Decay."
    )

    # 最大训练 epoch 数。
    #
    # 一个 epoch 通常表示完整遍历一次训练数据集。
    #
    # 例如：
    #   --epochs 20
    #
    # 表示最多训练 20 个 epoch。
    #
    # 实际训练是否提前结束，还可能受到：
    #   - Early Stopping
    #   - Trainer 配置
    #   - checkpoint 恢复状态
    # 等因素影响。
    parser.add_argument(
        "--epochs",
        type=int,
        help="Maximum number of epochs"
    )

    # Warmup Epoch 数量。
    #
    # Warmup 是指训练最开始的一段时间内，
    # 将学习率从较小值逐渐提升到目标学习率。
    #
    # 这样可以减少训练初期因为学习率过大导致的不稳定。
    #
    # 例如：
    #   --epochs 20
    #   --warmup 3
    #
    # 表示前 3 个 epoch 用于学习率 warmup。
    parser.add_argument(
        "--warmup",
        type=int,
        help="Number of warmup epochs"
    )

    # =========================
    # DataLoader 参数
    # =========================

    # DataLoader 的 worker 数量。
    #
    # worker 用于并行读取、解码和预处理训练图片。
    #
    # 例如：
    #   --nw 0
    #       所有数据读取都在主进程完成。
    #       最慢，但最适合排查 DataLoader 问题。
    #
    #   --nw 2
    #       使用 2 个子进程加载数据。
    #
    #   --nw 8
    #       使用 8 个子进程加载数据，吞吐量可能更高，
    #       但系统 RAM 占用也可能大幅增加。
    #
    # 特别注意：
    # num_workers 并不是越大越好。
    # 每个 worker 都可能提前读取 / 预处理 batch，
    # 所以大 batch + 多 worker 很容易导致 CPU 内存暴涨。
    #
    # 如果你的机器只有 32GB RAM，
    # 建议复现 BoQ 时先从：
    #   --nw 2
    # 开始。
    #
    # 如果仍然爆 RAM，可以测试：
    #   --nw 0
    #
    # 用来判断问题是不是 DataLoader 导致的。
    parser.add_argument(
        "--nw",
        type=int,
        help="Numbers of workers."
    )

    # =========================
    # Backbone 参数
    # =========================

    # Backbone 网络名称。
    #
    # Backbone 是负责从输入图片中提取视觉特征的主干网络。
    #
    # 当前代码支持：
    #   resnet50
    #   dinov2
    #
    # 示例：
    #
    #   python train.py --backbone resnet50
    #
    # 或：
    #
    #   python train.py --backbone dinov2
    #
    # DINOv2 通常使用 Transformer / ViT 架构，
    # 在视觉地点识别任务中能够提供较强的特征表达能力。
    #
    # 不同 backbone 对：
    #   - GPU 显存
    #   - CPU 内存
    #   - 训练速度
    #   - 输入图像尺寸
    # 都可能有不同影响。
    parser.add_argument(
        "--backbone",
        type=str,
        help="Backbone model name [resnet50, dinov2]"
    )

    # Backbone 中解冻的 block 数量。
    #
    # 在使用预训练 backbone 时，
    # 通常不会从头训练整个网络，而是冻结大部分参数，
    # 只对最后若干个 block 进行微调。
    #
    # 例如：
    #   --unfreeze_n 2
    #
    # 表示解冻 backbone 最后的 2 个 block。
    #
    # unfreeze_n 越大：
    #   + 可以让更多 backbone 参数适应当前任务
    #   + 理论上具有更强的微调能力
    #   - GPU 显存占用增加
    #   - 反向传播计算量增加
    #   - 训练速度变慢
    #   - 过拟合风险可能增加
    #
    # 如果显存比较紧张，可以适当减少该值。
    parser.add_argument(
        "--unfreeze_n",
        type=int,
        help="Number of blocks to unfreeze in the backbone."
    )

    # =========================
    # 特征维度参数
    # =========================

    # 最终输出描述子的维度（descriptor dimensionality）。
    #
    # 在图像检索 / Visual Place Recognition 中，
    # 模型通常会将一张图片编码成一个固定长度的向量，
    # 后续通过向量距离进行图片匹配和检索。
    #
    # 例如：
    #   --dim 4096
    #
    # 表示最终生成一个 4096 维的全局图像描述子。
    #
    # dim 越大：
    #   + 理论上能够保存更多特征信息
    #   - descriptor 占用更多内存 / 磁盘
    #   - 检索时计算量增加
    #   - BoQ 聚合层的参数量和显存占用也可能增加
    #
    # 如果目标是严格复现论文结果，
    # 建议保持论文 / 官方配置中的默认维度。
    parser.add_argument(
        "--dim",
        type=int,
        help="Output dimensionality."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()

    if args.seed:
        hparams.seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.bs:
        hparams.batch_size = args.bs
    if args.lr:
        hparams.lr = args.lr
    if args.wd:
        hparams.weight_decay = args.wd
    if args.epochs:
        hparams.max_epochs = args.epochs
    if args.warmup:
        hparams.warmup_epochs = args.warmup
    if args.nw:
        hparams.num_workers = args.nw
    if args.backbone:
        hparams.backbone_name = args.backbone
    if args.unfreeze_n:
        hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim:
        hparams.output_dim = args.dim

    train(hparams, dev_mode=args.dev)
