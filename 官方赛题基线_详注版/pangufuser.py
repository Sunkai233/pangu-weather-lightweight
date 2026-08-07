# =============================================================================
# 文件: pangufuser.py
# 作用: 主干融合块 —— 把若干个 EarthTransformer3DBlock(带地球位置偏置的 3D 窗口注意力块)堆叠起来,
#       在三维网格 (T/气压, 纬, 经) 上反复做窗口自注意力,融合多层高度、空间上的天气信息。
# 在网络中的位置: 整个网络的"脊柱"。embed 后的特征拉平成 [B, T*H*W, dim] 进来,经过 depth 层
#       Transformer 融合后再吐出同形状 [B, T*H*W, dim],交给上采样/PatchRecovery 还原成天气场。
# 关键点:
#   1) 每一层都是 3D 窗口注意力(Swin 风格): 只在小窗口内算 attention,省显存;窗口大小如 (2,6,12)。
#   2) shift_size 交替: 偶数层 (0,0,0)=普通窗口, 奇数层 None=移窗(SW-MSA)。移窗让相邻窗口之间也能交流信息,
#      否则信息永远被窗口边界挡住(这是 Swin Transformer 的精髓)。
#   3) drop_path 可以是标量或每层一个的序列(随机深度,越深的层丢得越多)。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from onescience.modules.transformer.onetransformer import OneTransformer


class PanguFuser(nn.Module):
    """
        Pangu-Weather 模型的三维特征融合模块，在给定三维网格上堆叠多层 3D Transformer 块以融合多时刻、多高度和空间信息。

        Args:
            dim (int): 输入与输出特征的通道维度
            input_resolution (tuple[int, int, int]): 三维输入特征的网格尺寸 (T, H, W)
            depth (int): 3D Transformer 块的层数
            num_heads (int): 多头自注意力的头数
            window_size (tuple[int, int, int]): 三维窗口注意力的窗口大小 (Wt, Wh, Ww)
            drop_path (float | Sequence[float]): DropPath / Stochastic Depth 比例或其序列
            mlp_ratio (float): 前馈网络隐藏层与特征维度的比例
            qkv_bias (bool): 是否在 QKV 投影中使用偏置
            qk_scale (float | None): QK 点积缩放因子
            drop (float): 特征上的 dropout 比例
            attn_drop (float): 注意力权重上的 dropout 比例
            norm_layer (nn.Module): 归一化层类型

        形状:
            输入:  x 形状为 (B, T * H * W, dim)，其中 (T, H, W) = input_resolution
            输出:  x 形状为 (B, T * H * W, dim)，与输入相同

        Example:
            >>> dim = 256
            >>> input_resolution = (10, 181, 360)
            >>> fuser = PanguFuser(
            ...     dim=dim,
            ...     input_resolution=input_resolution,
            ...     depth=4,
            ...     num_heads=8,
            ...     window_size=(2, 6, 12),
            ... )
            >>> B, T, H, W = 2, 10, 181, 360
            >>> x = torch.randn(B, T * H * W, dim)
            >>> out = fuser(x)
            >>> out.shape
            torch.Size([2, 10 * 181 * 360, 256])
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # 记录特征通道维、三维网格分辨率、堆叠层数,供 forward 和外部查询
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 堆 depth 个 3D Transformer 块。用 OneScience 的 OneTransformer 统一封装,
        # style="EarthTransformer3DBlock" 指定用"地球版"(带经纬对称位置偏置)的注意力实现。
        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer3DBlock",
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # 关键: 偶数层 shift=(0,0,0) 普通窗口; 奇数层 shift=None 走移窗(SW-MSA),交替进行
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    # drop_path 若是序列就按层取第 i 个,否则所有层共用同一个标量
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # 顺序穿过每一个 Transformer 块;每块内部含残差,进出形状始终是 (B, T*H*W, dim)
        for blk in self.blocks:
            x = blk(x)
        return x
