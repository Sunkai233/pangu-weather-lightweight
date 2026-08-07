# =============================================================================
# 文件: pangu.py
# 作用: Pangu-Weather 主模型——把 5 大积木按"U 形"串起来,完成 72通道输入 → 69通道预测
# 在流程中的位置: inference.py / train.py 里 model = Pangu(...) 就是这个类
# 关键点(★决赛必记):
#   1) 五大积木全靠工厂 OneXxx(style=...) 调用(来自 onescience.modules),自己不用手写实现:
#      OneEmbedding(切patch) / OneFuser(主干Transformer块) / OneSample(下/上采样) / OneRecovery(反卷积恢复)
#   2) U 形结构:embed → layer1 → 下采样 → layer2 → layer3 → 上采样 → layer4 → recovery,
#      并且有一条 skip:layer1 的输出直接 concat 到 layer4 输出(编码器↔解码器跳连)。
#   3) 地面场走 2D 分支,高空场走 3D 分支,在 patch 之后沿"层"维拼成一个 3D 张量统一处理。
#   4) 输入 72 = 4 地面 + 3 静态 + 65 高空;输出 = (地面 4通道, 高空 5×13)。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================

import math
import torch
import numpy as np

from torch import nn
from dataclasses import dataclass
from onescience.models.meta import ModelMetaData

# ★五大积木工厂:style 字符串 → 具体实现类(实现都在 onescience.modules 里,可 less 源码)
from onescience.modules import (
    OneEmbedding,   # Patch Embedding(切patch+投影)
    OneFuser,       # 主干融合块(堆叠 3D Transformer 窗口注意力)
    OneRecovery,    # Patch Recovery(反卷积恢复分辨率)
    OneSample,      # 下采样/上采样
)

@dataclass
class MetaData(ModelMetaData):
    # 模型元信息(声明支持哪些优化/推理特性,框架据此决定行为)
    name: str = "Pangu"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict          # 不支持 JIT(ONNX 算子冲突)
    cuda_graphs: bool = True                          # 支持 CUDA Graphs
    amp: bool = True                                  # 支持混合精度
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU        # CPU 上没有 FFT 算子,不支持
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Pangu(nn.Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),        # 全球网格:纬 721 × 经 1440
        patch_size=(2, 4, 4),        # patch 大小 (气压层, 纬, 经)
        embed_dim=192,               # 嵌入维度
        num_heads=(6, 12, 12, 6),    # 四个阶段各自的注意力头数
        window_size=(2, 6, 12),      # 三维窗口大小
    ):
        super().__init__()
        # DropPath(随机深度)比率:8 个块从 0 线性增到 0.2(越深丢得越多,正则化)
        drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        # (地形/陆海/土壤三个静态掩码在输入端就拼进 72 通道了,见 inference.py)

        # ── 两个 Patch Embedding:地面场用 2D,高空场用 3D ──
        self.patchembed2d = OneEmbedding(style="PanguEmbedding2D")   # 输入地面 [B,7,721,1440] → [B,192,181,360]
        self.patchembed3d = OneEmbedding(style="PanguEmbedding3D")   # 输入高空 [B,5,13,721,1440] → [B,192,7,181,360]

        # patch 之后的三维网格尺寸:层维=8(1地面+7高空),纬=⌈721/4⌉=181,经=⌈1440/4⌉=360
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        # layer1:第一个主干融合块(编码器第 1 段),深度 2,不改分辨率
        self.layer1 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        # 下采样后的网格:纬经各减半(181→91,360→180),层维不变
        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )

        # 下采样:把空间分辨率降一半、通道翻倍(192→384),减计算量
        self.downsample = OneSample(
            style="PanguDownSample3D",
            in_dim=embed_dim,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )
        # layer2:低分辨率主干块,深度 6,通道 384
        self.layer2 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        # layer3:同分辨率再来一段深度 6(编码/解码中段主力)
        self.layer3 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        # 上采样:分辨率还原、通道减半(384→192),回到高分辨率
        self.upsample = OneSample(
            style="PanguUpSample3D",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_inp_shape_downsample,
            output_resolution=patched_inp_shape
        )
        # layer4:高分辨率主干块(解码器最后一段),深度 2
        self.layer4 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        # ↑ 论文原话:编码器第2层输出 与 解码器第7层输出 沿通道拼接(就是下面 forward 里的 skip concat)
        # ── 两个 Patch Recovery:把 patch 特征反卷积回原分辨率 ──
        # 注意 recovery 的输入通道是 2*embed_dim,因为 skip 拼接后通道翻倍
        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery2D"      # 恢复地面 4 通道 [B,4,721,1440]
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery3D"      # 恢复高空 5 通道 [B,5,13,721,1440]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        # x 的 72 通道 = [0:7]=4地面+3静态 给 2D 分支;[7:]=65高空 给 3D 分支
        surface = x[:, :7, :, :] # 1, 72, 721, 1440         # 地面+静态,共 7 通道
        # 高空 65 通道 reshape 成 (5 变量, 13 层),变成 3D 体数据
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3]) # torch.Size([1, 5, 13, 721, 1440])
        surface = self.patchembed2d(surface) # torch.Size([1, 192, 181, 360])          # 地面切 patch
        upper_air = self.patchembed3d(upper_air) #torch.Size([1, 192, 7, 181, 360])     # 高空切 patch(层 13→7)

        # 把地面(补一个层维=1)和高空(层维=7)沿"层"维拼成 8 层的统一 3D 特征
        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2) # torch.Size([1, 192, 8, 181, 360])
        B, C, Pl, Lat, Lon = x.shape
        # 展平成 token 序列 [B, N, C](N = 8*181*360 = 521280),喂给 Transformer
        x = x.reshape(B, C, -1).transpose(1, 2) # torch.Size([1, 521280, 192])

        x = self.layer1(x)      # 编码器第 1 段

        skip = x                # ★保存 skip:layer1 输出,留给最后和 layer4 输出拼接

        x = self.downsample(x)  # 下采样(分辨率减半、通道翻倍)
        x = self.layer2(x)      # 低分辨率段 1
        x = self.layer3(x)      # 低分辨率段 2
        x = self.upsample(x)    # 上采样(还原分辨率)
        x = self.layer4(x)      # 解码器最后一段

        output = torch.concat([x, skip], dim=-1)   # ★U 形跳连:沿通道拼 skip,通道 192→384
        # 还原成 3D 网格 [B, 2C, Pl, Lat, Lon]
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]      # 第 0 层 = 地面分支
        output_upper_air = output[:, :, 1:, :, :]   # 第 1~7 层 = 高空分支

        # 反卷积恢复到原始物理分辨率
        output_surface = self.patchrecovery2d(output_surface)       # [B,4,721,1440]
        output_upper_air = self.patchrecovery3d(output_upper_air)   # [B,5,13,721,1440]
        return output_surface, output_upper_air     # 返回(地面预测, 高空预测)
