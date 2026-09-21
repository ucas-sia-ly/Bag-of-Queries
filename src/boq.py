# ----------------------------------------------------------------------------
# 版权声明分隔线：这一行本身不参与程序执行，仅用于让文件头更清晰。
# Copyright (c) 2024 Amar Ali-bey
# 上一行说明该源码版权归 Amar Ali-bey 所有，版权年份为 2024。
#
# 空注释行：仅用于排版，使版权信息、项目地址和许可证说明之间更易阅读。
# https://github.com/amaralibey/Bag-of-Queries
# 上一行给出了 Bag-of-Queries（BoQ）项目的 GitHub 地址，可用于查看原项目、README、训练代码等。
#
# 空注释行：仍然只是文件头排版，不会影响 Python 程序。
# See LICENSE file in the project root.
# 上一行提示：具体的软件许可证条款请查看项目根目录中的 LICENSE 文件。
# ----------------------------------------------------------------------------
# 文件头结束分隔线，同样只用于排版，不参与运行。

# 导入 PyTorch 顶层包；后续神经网络层、张量运算、参数定义等全部通过 torch 使用。
import torch


# 定义一个 BoQBlock 模块。
# 继承 torch.nn.Module 后，该类才能被 PyTorch 正确注册参数、切换 train()/eval() 模式、移动到 GPU 等。
class BoQBlock(torch.nn.Module):
    # BoQBlock 的构造函数。
    # in_dim：每个 token / 特征向量的维度 D。
    # num_queries：可学习 Query 的数量 Q。
    # nheads：多头注意力的头数，默认值为 8；要求 in_dim 能被 nheads 整除。
    def __init__(self, in_dim, num_queries, nheads=8):
        # 调用父类 torch.nn.Module 的构造函数。
        # 这一步非常重要，否则该模块内部创建的 Parameter、子 Module 等无法被 PyTorch 正确管理。
        super(BoQBlock, self).__init__()

        # 创建一个标准 TransformerEncoderLayer，用于先对输入 token 序列做一次 Transformer 编码。
        # d_model=in_dim：每个 token 的特征维度为 D=in_dim。
        # nhead=nheads：使用 nheads 个注意力头，每个头通常处理 D/nheads 维特征。
        # dim_feedforward=4*in_dim：Transformer 中前馈网络隐藏层维度设置成 4D，这是常见配置。
        # batch_first=True：规定输入和输出张量格式为 [B, N, D]，而不是默认的 [N, B, D]。
        # dropout=0.：将该 EncoderLayer 内部 dropout 关闭，即 dropout 概率为 0。
        # 若输入 x 的形状是 [B, N, D]，该层输出形状仍然是 [B, N, D]。
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)

        # 创建一组“可学习的 Query 向量”，其初始值由标准正态分布 torch.randn 随机生成。
        # 初始形状为 [1, Q, D]：
        #   第 0 维的 1 表示先只保存一份 Query 模板；forward 时再复制到 batch 中的每个样本。
        #   第 1 维 num_queries=Q 表示 Query 的数量。
        #   第 2 维 in_dim=D 表示每个 Query 的特征维度。
        # torch.nn.Parameter 会把该张量注册为模型可训练参数，因此反向传播时它会得到梯度并被优化器更新。
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # 原作者说明：下面的 self-attention 和归一化主要用于训练阶段。
        # 因为这部分只依赖可学习 queries、本身不依赖当前输入图像 x，所以在 eval 推理阶段模型参数固定后，
        # 可以预先计算并缓存处理后的 Query，以减少重复计算。
        # the following two lines are used during training only, you can cache their output in eval.

        # 为 Query 创建多头“自注意力”层。
        # embed_dim=in_dim：Query 的嵌入维度为 D。
        # num_heads=nheads：将 D 维特征拆成多个注意力头并行计算。
        # batch_first=True：输入输出都采用 [B, Q, D] 格式。
        # forward 中会把 q 同时作为 query、key、value，因此属于 self-attention。
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        # 创建 LayerNorm，用于对每个 Query 最后一维 D 做归一化。
        # 输入若为 [B, Q, D]，LayerNorm(in_dim) 会对每个 [D] 特征向量独立归一化，输出形状不变。
        # 这里用于稳定 Query 自注意力之后的数值分布和训练过程。
        self.norm_q = torch.nn.LayerNorm(in_dim)

        # 原代码中的视觉分隔注释，不参与程序运行。
        #####

        # 创建 Query 与输入特征 x 之间的“交叉注意力（cross-attention）”层。
        # 在 forward 中：q 作为 Query，x 同时作为 Key 和 Value。
        # 因此每个可学习 Query 会根据注意力权重，从输入 token 序列 x 中聚合自己最关注的信息。
        # 输入大致为 q:[B,Q,D]、x:[B,N,D]，输出 out:[B,Q,D]。
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        # 对 cross-attention 的输出做 LayerNorm。
        # 仍然只对最后的 D 维进行归一化，因此不会改变 [B, Q, D] 的张量形状。
        self.norm_out = torch.nn.LayerNorm(in_dim)

    # 定义该模块的前向传播过程。
    # x 预期形状为 [B, N, D]：
    #   B = batch size；
    #   N = 输入 token 数量，例如图像展平后 N=H×W；
    #   D = in_dim，即每个 token 的特征维度。
    def forward(self, x):
        # 读取输入 x 的第 0 维大小，也就是 batch size B。
        # 例如 x.shape=[16, 196, 512] 时，B=16。
        B = x.size(0)

        # 将输入 token 序列送入 TransformerEncoderLayer。
        # Encoder 内部会执行自注意力、前馈网络、残差连接和归一化等标准 Transformer 操作。
        # 输入 x:[B,N,D]，输出仍为 x:[B,N,D]。
        # 这里相当于先让所有输入空间 token 彼此进行信息交互，再交给 Query 聚合。
        x = self.encoder(x)

        # 将共享的可学习 Query 从 [1,Q,D] 沿 batch 维复制 B 份，得到 [B,Q,D]。
        # repeat(B,1,1) 会真实复制张量数据；每个 batch 样本使用相同的一组可学习 Query 参数。
        # 虽然每个样本最开始的 q 数值相同，但后续 cross-attention 会因为各自的输入 x 不同而产生不同输出。
        q = self.queries.repeat(B, 1, 1)

        # 原作者说明：下面两行主要用于训练阶段。
        # the following two lines are used during training.
        # 原作者指出这样做是为了增强训练稳定性。
        # for stability purposes

        # 对 Query 自身做多头自注意力，并通过残差连接加回原始 q。
        # self.self_attn(q, q, q) 中：
        #   第一个 q 是 attention 的 Query；
        #   第二个 q 是 Key；
        #   第三个 q 是 Value。
        # MultiheadAttention 返回一个元组：(attention_output, attention_weights)。
        # [0] 只取 attention_output，其形状为 [B,Q,D]，忽略这里的 Query 自注意力权重。
        # 最外层 q + ... 构成残差连接，使结果等于“原始 Query + Query 间交互信息”。
        q = q + self.self_attn(q, q, q)[0]

        # 对经过自注意力和残差连接后的 Query 做 LayerNorm。
        # 输入和输出形状都是 [B,Q,D]；该操作有助于稳定特征尺度和训练梯度。
        q = self.norm_q(q)

        # 原代码中的视觉分隔注释，不参与运行。
        #######

        # 执行交叉注意力：q 是 Query，x 是 Key，x 也是 Value。
        # 直观理解：每个可学习 Query 都会“询问”所有输入 token，并根据相关性加权汇总输入信息。
        # q 的形状为 [B,Q,D]，x 的形状为 [B,N,D]。
        # out：每个 Query 聚合得到的特征，形状为 [B,Q,D]。
        # attn：注意力权重。PyTorch 默认 average_attn_weights=True，因此通常形状为 [B,Q,N]，
        #       表示每个样本的每个 Query 对 N 个输入 token 的平均多头注意力分布。
        out, attn = self.cross_attn(q, x, x)

        # 对交叉注意力输出 out 做 LayerNorm。
        # 形状仍保持 [B,Q,D]，用于规范化每个 Query 聚合得到的 D 维描述向量。
        out = self.norm_out(out)

        # 返回三个结果：
        # 1) x：经过当前 TransformerEncoderLayer 编码后的输入 token，[B,N,D]；
        #       外层 BoQ 会把它继续传给下一个 BoQBlock，实现多层级联。
        # 2) out：当前 BoQBlock 的 Q 个 Query 聚合结果，[B,Q,D]；后续会收集各层 out 并拼接。
        # 3) attn.detach()：交叉注意力权重，通常为 [B,Q,N]；detach() 将它从 autograd 计算图中分离，
        #       因此外部若只拿它做可视化/分析，不会额外保留这条分支的反向传播图，也不会通过它传播梯度。
        return x, out, attn.detach()


# 定义完整的 BoQ（Bag of Queries）聚合模块。
# 该模块接收 CNN/ResNet 一类网络输出的二维特征图，先降维，再展平成 token 序列，
# 然后连续通过多个 BoQBlock，最终生成一个固定长度、L2 归一化的全局描述向量。
class BoQ(torch.nn.Module):
    # BoQ 构造函数。
    # in_channels=1024：输入二维特征图的通道数 C_in，默认 1024。
    # proj_channels=512：通过 3×3 卷积投影后的通道数 D，默认 512；同时也是后续 Transformer 的 token 维度。
    # num_queries=32：每个 BoQBlock 中可学习 Query 的数量 Q。
    # num_layers=2：串联多少个 BoQBlock，记为 L。
    # row_dim=32：最终线性层把“所有层的 Query 维度”压缩到的列/行特征维度 R。
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        # 调用 torch.nn.Module 父类构造函数。
        # 这里使用 Python 3 的简写 super().__init__()，作用与前面 super(BoQBlock, self).__init__() 等价。
        super().__init__()

        # 创建一个二维 3×3 卷积，用于把输入特征图的通道数从 in_channels 投影到 proj_channels。
        # kernel_size=3：使用 3×3 卷积核，可同时融合局部邻域信息。
        # padding=1：四周补 1 个像素，因此 stride 默认为 1 时空间尺寸 H、W 不变。
        # 若输入为 [B,C_in,H,W]，输出将是 [B,D,H,W]，其中 D=proj_channels。
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)

        # 为卷积投影后的 token 特征创建 LayerNorm。
        # 后续会先把特征排列为 [B,H×W,D]，因此 LayerNorm(proj_channels) 正好对最后一维 D 归一化。
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        # 把后续 BoQBlock 使用的 token 特征维度记为 in_dim。
        # 因为输入已经通过卷积投影到 proj_channels，所以 in_dim = proj_channels。
        in_dim = proj_channels

        # 创建 num_layers 个 BoQBlock，并使用 ModuleList 注册它们。
        # ModuleList 与普通 Python list 的关键区别是：其中的子模块会被 PyTorch 正确登记，
        # 因而它们的参数会出现在 model.parameters() / state_dict() 中，也会跟随 .to(device) 移动设备。
        self.boqs = torch.nn.ModuleList([
            # 对每一层构造一个 BoQBlock：token 维度为 in_dim，Query 数量为 num_queries。
            # nheads=in_dim//64：注意力头数按“每约 64 个通道一个头”的方式设置。
            # 例如 in_dim=512 时，nheads=512//64=8。
            # 注意：MultiheadAttention 要求 embed_dim 能被 num_heads 整除；此外若 in_dim<64，整数除法会得到 0，配置会无效。
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])

        # 创建最终的全连接层。
        # 输入维度为 num_layers*num_queries = L×Q。
        # 输出维度为 row_dim = R。
        # 注意这里 Linear 会作用在输入张量的“最后一维”上。
        # 后续 out 会被整理成 [B,D,L×Q]，因此 fc 会把最后一维 L×Q 映射为 R，得到 [B,D,R]。
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)

    # 定义完整 BoQ 的前向传播。
    # x 预期是 CNN/ResNet 输出的二维特征图，形状一般为 [B,C_in,H,W]。
    def forward(self, x):
        # 原作者说明：使用 ResNet 等骨干网络时，通过 3×3 卷积减少输入通道维度。
        # reduce input dimension using 3x3 conv when using ResNet

        # 对输入二维特征图做 3×3 卷积投影。
        # 输入：[B,C_in,H,W]。
        # 输出：[B,D,H,W]，其中 D=proj_channels；由于 padding=1、stride=1，H 和 W 保持不变。
        x = self.proj_c(x)

        # 先执行 x.flatten(2)：从第 2 维开始把 H、W 两个空间维度展平，
        # [B,D,H,W] -> [B,D,H×W]。
        # 再执行 permute(0,2,1)：交换第 1、2 维，
        # [B,D,H×W] -> [B,H×W,D]。
        # 这样每个空间位置就变成一个 token，总 token 数 N=H×W，每个 token 的维度为 D。
        x = x.flatten(2).permute(0, 2, 1)

        # 对每个空间 token 的 D 维特征做 LayerNorm。
        # 输入和输出形状都是 [B,N,D]，其中 N=H×W。
        # 归一化后的 token 序列随后送入多个 BoQBlock。
        x = self.norm_input(x)

        # 创建空列表 outs，用于保存每一个 BoQBlock 输出的 Query 聚合特征 out。
        # 每个元素的形状通常都是 [B,Q,D]。
        outs = []

        # 创建空列表 attns，用于保存每一个 BoQBlock 的 cross-attention 权重。
        # 每个元素通常为 [B,Q,N]，可用于分析/可视化每个 Query 关注了哪些输入位置。
        attns = []

        # 依次遍历所有 BoQBlock。
        # len(self.boqs)=num_layers，因此 i 的范围是 0 到 num_layers-1。
        for i in range(len(self.boqs)):
            # 调用第 i 个 BoQBlock。
            # 输入 x:[B,N,D]。
            # 返回的新 x 仍是 [B,N,D]，但已经被当前层 TransformerEncoderLayer 进一步编码；
            # out 是当前层的 Query 聚合结果 [B,Q,D]；
            # attn 是当前层的交叉注意力权重，通常为 [B,Q,N]。
            # 新的 x 会进入下一层 BoQBlock，因此各层是串联关系，而不是彼此独立并行。
            x, out, attn = self.boqs[i](x)

            # 把当前层的 Query 输出 out 保存到 outs。
            # 循环结束后，outs 中一共有 L=num_layers 个 [B,Q,D] 张量。
            outs.append(out)

            # 把当前层的交叉注意力权重保存到 attns。
            # 这样最终不仅得到全局描述向量，还能拿到每一层的注意力图进行解释或可视化。
            attns.append(attn)

        # 沿 Query 所在的第 1 维拼接所有层的输出。
        # 每层 out:[B,Q,D]，一共有 L 层，因此拼接后：
        # [B,Q,D] × L -> [B,L×Q,D]。
        # 这一步把所有 BoQBlock 产生的 Query 描述集中到一个张量中。
        out = torch.cat(outs, dim=1)

        # 先执行 out.permute(0,2,1)： 
        # [B,L×Q,D] -> [B,D,L×Q]。
        # 然后送入 self.fc。因为 torch.nn.Linear 总是作用于最后一维，
        # 所以它把最后的 L×Q 维映射为 row_dim=R：
        # [B,D,L×Q] -> [B,D,R]。
        # 可以理解为：对每个特征通道 D，学习如何组合所有层的所有 Query 响应。
        out = self.fc(out.permute(0, 2, 1))

        # 从第 1 维开始把 [D,R] 两个维度展平成一个维度。
        # [B,D,R] -> [B,D×R]。
        # 因此每个输入样本最终得到一个固定长度的全局描述向量，长度为 proj_channels*row_dim。
        # 默认 D=512、R=32 时，描述子长度为 512×32=16384。
        out = out.flatten(1)

        # 对最终描述向量沿最后一维做 L2 归一化（p=2）。
        # 对每个样本，归一化后向量的欧氏范数约等于 1。
        # 这在图像检索、地点识别、特征匹配等任务中很常见，便于使用余弦相似度或欧氏距离比较描述子。
        # 输入/输出形状都为 [B,D×R]。
        out = torch.nn.functional.normalize(out, p=2, dim=-1)

        # 返回最终的全局描述向量 out，以及所有 BoQBlock 的注意力权重列表 attns。
        # out：[B,D×R]，且每个样本已经做 L2 归一化。
        # attns：长度为 L 的 Python 列表，每个元素通常为 [B,Q,N]。
        return out, attns
