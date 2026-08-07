# =============================================================================
# 文件: pangupatchrecovery3d.py
# 作用: 3D Patch Recovery —— 3D Embedding 的逆操作。用三维反卷积把 patch 特征放大回
#       原始的(气压层数, 纬, 经)分辨率,恢复出高空场,再裁掉当初补的边。
# 在网络中的位置: 高空场的输出端。输入 [B, in_chans, 7, 181, 360] ——> 输出 [B, 5, 13, 721, 1440]。
#       out_chans=5 对应 5 个高空变量(z/q/t/u/v),每个都在 13 个气压层上。
# 关键点:
#   1) ConvTranspose3d(kernel=stride=patch_size=(2,4,4)) 把每个立方 patch 特征展开成 2x4x4 的体素块。
#   2) 放大后是补齐尺寸(14x724x1440),要 center-crop 回 13x721x1440。
#   3) 三个方向裁边量与 3D Embedding 补零对称: 气压多 1(前裁0后裁1)、纬多 3(上1下2)、经不多。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
import torch
from torch import nn

class PanguPatchRecovery3D(nn.Module):

    """
        Pangu-Weather 模型中的三维 Patch 恢复模块，用三维反卷积将 Patch 特征还原为原始层数与空间分辨率的三维场，并裁剪掉补零边界。

        Args:
            img_size (tuple[int, int, int]): 输出目标场的尺寸 (L, H, W)，分别对应垂直层数和空间网格
            patch_size (tuple[int, int, int]): 三维 Patch 大小 (patch_l, patch_h, patch_w)，即反卷积的 kernel_size 与 stride
            in_chans (int): 输入特征通道数
            out_chans (int): 输出场通道数

        形状:
            输入:  x 形状为 (B, in_chans, L', H', W')
            输出:  y 形状为 (B, out_chans, img_size[0], img_size[1], img_size[2])

        Example:
            >>> recovery3d = PanguPatchRecovery3D(
            ...     img_size=(13, 721, 1440),
            ...     patch_size=(2, 4, 4),
            ...     in_chans=384,
            ...     out_chans=5,
            ... )
            >>> x = torch.randn(2, 384, 7, 181, 360)
            >>> y = recovery3d(x)
            >>> y.shape
            torch.Size([2, 5, 13, 721, 1440])
    """
    def __init__(self, img_size = (13, 721, 1440),
                 patch_size = (2, 4, 4),
                 in_chans = 192*2,
                 out_chans = 5):
        super().__init__()
        # 记录目标三维尺寸 (13, 721, 1440),裁边时用
        self.img_size = img_size
        # 三维转置卷积: 每个 patch 特征放大成 2(气压)x4(纬)x4(经)的体素块,通道降到 out_chans=5
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        # 反卷积放大: [B,in_chans,7,181,360] -> [B,5,14,724,1440] (7*2=14, 181*4=724, 360*4=1440)
        output = self.conv(x)
        # 取放大后的气压层 Pl=14、纬 Lat=724、经 Lon=1440
        _, _, Pl, Lat, Lon = output.shape

        # 各方向比目标多出的量: 气压 14-13=1, 纬 724-721=3, 经 1440-1440=0
        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        # 气压方向对半裁: 前裁 0、后裁 1(与 Embedding 前0后1补零对称)
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        # 纬度方向对半裁: 上裁 1、下裁 2
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        # 经度方向对半裁(本例为 0)
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        # center-crop: 三个方向都切掉多补的边,还原成干净的 13x721x1440 高空场
        return output[
            :,
            :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]




