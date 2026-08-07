# =============================================================================
# 文件: panguembedding2d.py
# 作用: 2D Patch Embedding —— 把"地面场"(msl/u10/v10/t2m + land_mask/soil_type/topography,共 4+3=7 通道)
#       切成一个个不重叠的小方块(patch),每个 patch 用一个卷积核压成一个 embed_dim 维的特征向量。
# 在网络中的位置: 地面场的输入端。输入 [B, 7, 721, 1440] ——> 输出 [B, 192, 181, 360]。
#       与之并列的是 panguembedding3d.py(处理 65 个高空通道)。两条支路 embed 后在主干里拼接融合。
# 关键点:
#   1) "切 patch + 线性投影" 本质就是一个 stride=patch_size 的 Conv2d(kernel 也等于 patch_size,块与块不重叠)。
#   2) 721 不能被 4 整除,所以先 ZeroPad 补齐到能整除;补法是"前后对半补",高度 721->724(上补1、下补2)。
#   3) 输出的空间尺寸: H 721->⌈721/4⌉=181, W 1440->1440/4=360。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
import torch
from torch import nn

class PanguEmbedding2D(nn.Module):
    """
        将二维图像分割为不重叠的 patch 并嵌入到向量空间。

        Args:
            img_size (tuple[int, int]): 输入图像尺寸 (H, W)
            patch_size (tuple[int, int]): 每个 patch 的大小 (patch_h, patch_w)
            in_chans (int): 输入图像通道数
            embed_dim (int): 每个 patch 嵌入后的向量维度
            norm_layer (nn.Module, optional): 归一化层，默认为 None。常用: nn.LayerNorm

        形状:
            输入: (B, C, H, W)
            输出: (B, embed_dim, H', W')，其中 H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉

        Example:
            >>> patch_embed = PatchEmbed2D(
            ...     img_size=(128, 256),
            ...     patch_size=(4, 4),
            ...     in_chans=3,
            ...     embed_dim=96
            ... )
            >>> x = torch.randn(8, 3, 128, 256)
            >>> out = patch_embed(x)
            >>> out.shape
            torch.Size([8, 96, 32, 64])
    """

    def __init__(self, img_size=(721, 1440),
                    patch_size=(4, 4),
                    embed_dim=192,
                    in_chans = 4+3,
                    norm_layer=None,
                    ):

        super().__init__()
        # 拆出输入分辨率: 地面场高(纬向)=721, 宽(经向)=1440
        height, width = img_size
        # 拆出 patch 的高和宽,默认 4x4(注意变量名 w_path_size 是官方原始拼写,勿改)
        h_patch_size, w_path_size = patch_size
        # 卷积步长 = patch 大小,保证 patch 之间不重叠(切块效果)
        stride = patch_size
        # 四个方向的补零量先都初始化为 0
        padding_left = padding_right = padding_top = padding_bottom = 0
        # 高度除以 patch 高之后的余数: 721 % 4 = 1 (说明不能整除,需要补齐)
        h_remainder = height % h_patch_size
        # 宽度除以 patch 宽之后的余数: 1440 % 4 = 0 (能整除,不用补)
        w_remainder = width % w_path_size

        # 高度方向需要补齐: 补到能被 patch 高整除,补的总量前后对半分
        if h_remainder:
            # 还差多少才能凑齐一个 patch: 4-1=3
            h_pad = h_patch_size - h_remainder
            # 上边补一半(向下取整): 3//2=1
            padding_top = h_pad // 2
            # 下边补剩下的: 3-1=2 (前后对半、下边略多)
            padding_bottom = int(h_pad - padding_top)

        # 宽度方向同理(本例 1440 能整除,不进这个分支)
        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        # 用 0 值把上下左右补齐(顺序是 左,右,上,下),高度 721->724
        self.pad = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )
        # 核心: kernel=stride=patch_size 的卷积 = 每个 patch 拉平做一次线性投影 -> embed_dim 维
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        # 可选的归一化层(默认 None)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        # 输入形状: B=批大小, C=7(4地面+3静态掩码), H=721, W=1440
        B, C, H, W = x.shape
        # 先补零对齐: [B,7,721,1440] -> [B,7,724,1440]
        x = self.pad(x)
        # 切 patch + 投影: [B,7,724,1440] -> [B,192,181,360] (724/4=181, 1440/4=360)
        x = self.proj(x)
        # 若有归一化: 先把通道维换到最后 (B,H',W',C) 做 LayerNorm,再换回 (B,C,H',W')
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
