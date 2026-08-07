# =============================================================================
# 文件: panguembedding3d.py
# 作用: 3D Patch Embedding —— 把"高空场"切成三维小块。高空有 5 个变量(z/q/t/u/v),
#       每个变量在 13 个气压层上都有一张 721x1440 的图,所以是一个 [B, 5, 13, 721, 1440] 的五维张量。
#       这里在 (气压层, 纬, 经) 三个方向同时切 patch,每个立方体 patch 压成一个 embed_dim 维向量。
# 在网络中的位置: 高空场的输入端。输入 [B, 5, 13, 721, 1440] ——> 输出 [B, 192, 7, 181, 360]。
#       与 panguembedding2d.py(地面场)并列,两支 embed 后在主干融合。
# 关键点:
#   1) 和 2D 一样,"切 patch+线性投影" 就是一个 stride=patch_size 的 Conv3d(kernel=stride,块不重叠)。
#   2) 三个方向都要补齐到能整除 patch: 气压 13->14(前补0后补1)、纬 721->724(上1下2)、经 1440 整除不补。
#   3) 输出尺寸: 气压 13->⌈13/2⌉=7, 纬 721->181, 经 1440->360。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
import torch
from torch import nn

class PanguEmbedding3D(nn.Module):
    """
        将三维图像分割为不重叠的 patch 并嵌入到向量空间。

        Args:
            img_size (tuple[int, int, int]): 输入图像尺寸 (P, H, W)
            patch_size (tuple[int, int, int]): 每个 patch 的大小 (patch_p, patch_h, patch_w)
            in_chans (int): 输入图像通道数
            embed_dim (int): 每个 patch 嵌入后的向量维度
            norm_layer (nn.Module, optional): 归一化层，默认为 None

        形状:
            输入: (B, C, P, H, W)
            输出: (B, embed_dim, P', H', W'), 其中 P' = ⌈P / patch_p⌉, H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉

        Example:
            >>> patch_embed = PatchEmbed3D(
            ...     img_size=(13, 128, 256),
            ...     patch_size=(1, 4, 4),
            ...     in_chans=5,
            ...     embed_dim=192
            ... )
            >>> x = torch.randn(4, 5, 13, 128, 256)
            >>> out = patch_embed(x)
            >>> out.shape
            torch.Size([4, 192, 13, 32, 64])

    """

    def __init__(self,
                img_size = (13, 721, 1440),
                patch_size = (2, 4, 4),
                in_chans = 5,
                embed_dim = 192,
                norm_layer = None):

        super().__init__()

        # 拆出输入三维尺寸: level=13(气压层数), height=721(纬), width=1440(经)
        level, height, width = img_size
        # 拆出 patch 三维尺寸: 气压方向 2 层一块, 纬/经 各 4 格一块
        l_patch_size, h_patch_size, w_patch_size = patch_size

        # 六个方向补零量初始化为 0 (前后=气压, 上下=纬, 左右=经)
        padding_left = (
            padding_right
        ) = padding_top = padding_bottom = padding_front = padding_back = 0

        # 三个方向分别算余数,判断是否需要补齐到整除 patch
        l_remainder = level % l_patch_size      # 13 % 2 = 1 (需补)
        h_remainder = height % h_patch_size     # 721 % 4 = 1 (需补)
        w_remainder = width % w_patch_size      # 1440 % 4 = 0 (不补)

        # 气压层方向补齐: 差 1 层,前补 0、后补 1 -> 13 变 14
        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        # 纬度方向补齐: 差 3,上补 1、下补 2 -> 721 变 724
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        # 经度方向(本例整除,不进此分支)
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        # 三维补零(顺序: 左右, 上下, 前后),把 [.,.,13,721,1440] 补成 [.,.,14,724,1440]
        self.pad = nn.ZeroPad3d(
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            )
        )
        # 核心: kernel=stride=patch_size 的三维卷积 = 每个立方体 patch 做一次线性投影 -> embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # 可选归一化(默认 None)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        # 输入: B=批, C=5(z/q/t/u/v), L=13(气压层), H=721, W=1440
        B, C, L, H, W = x.shape
        # 补零对齐: [B,5,13,721,1440] -> [B,5,14,724,1440]
        x = self.pad(x)
        # 三维切 patch+投影: -> [B,192,7,181,360] (14/2=7, 724/4=181, 1440/4=360)
        x = self.proj(x)
        # 若有归一化: 把通道维 permute 到最后做 LayerNorm 再换回 (注意这里判断的是 self.norm 是否为真)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x
