# =============================================================================
# 文件: pangupatchrecovery2d.py
# 作用: 2D Patch Recovery —— Embedding 的逆操作。用反卷积(转置卷积)把每个 patch 特征
#       重新"放大"回原始像素分辨率,恢复出地面场,再把当初为对齐补的边裁掉(center-crop)。
# 在网络中的位置: 地面场的输出端。输入 [B, in_chans, 181, 360] ——> 输出 [B, 4, 721, 1440]。
#       out_chans=4 对应 4 个地面预报量(msl/u10/v10/t2m)。in_chans 默认 192*2 是因为主干输出会拼接。
# 关键点:
#   1) ConvTranspose2d(kernel=stride=patch_size) 是 Conv2d 切 patch 的逆: 每个 patch 特征展开成 patch_h x patch_w 像素块。
#   2) 反卷积得到的是补齐后的大小(724x1440),末尾要 center-crop 掉当初补的边,还原到 721x1440。
#   3) crop 的补边量与 Embedding 补零一致: 高度多出 3 -> 上裁 1、下裁 2;宽度不多不裁。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
import torch
from torch import nn


class PanguPatchRecovery2D(nn.Module):

    """
        Pangu-Weather 模型中的二维 Patch 恢复模块，用反卷积将 Patch 特征还原为原始空间分辨率的二维场，并裁剪掉补零边界。

        Args:
            img_size (tuple[int, int]): 输出目标图像尺寸 (H, W)
            patch_size (tuple[int, int]): Patch 大小 (patch_h, patch_w)，即反卷积的 kernel_size 与 stride
            in_chans (int): 输入特征通道数
            out_chans (int): 输出图像通道数

        形状:
            输入:  x 形状为 (B, in_chans, H', W')
            输出:  y 形状为 (B, out_chans, img_size[0], img_size[1])

        Example:
            >>> recovery = PanguPatchRecovery2D(
            ...     img_size=(721, 1440),
            ...     patch_size=(4, 4),
            ...     in_chans=384,
            ...     out_chans=4,
            ... )
            >>> x = torch.randn(2, 384, 181, 360)
            >>> y = recovery(x)
            >>> y.shape
            torch.Size([2, 4, 721, 1440])
    """
    def __init__(self,
                img_size = (721, 1440),
                patch_size = (4, 4),
                in_chans = 192*2,
                out_chans = 4):
        super().__init__()
        # 记录最终要还原到的目标分辨率 (721, 1440),forward 里裁边时用
        self.img_size = img_size
        # 转置卷积: kernel=stride=patch_size,把每个 patch 特征放大成 4x4 的像素块,通道降到 out_chans
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        # 反卷积放大: [B,in_chans,181,360] -> [B,4,724,1440] (181*4=724, 360*4=1440)
        output = self.conv(x)
        # 取出放大后的高 H=724、宽 W=1440
        _, _, H, W = output.shape
        # 比目标多出来的像素(即当初 Embedding 补的边): 高多 724-721=3, 宽多 1440-1440=0
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        # 把多出来的高对半裁: 上裁 1、下裁 2 (与 Embedding 的上1下2补零对称)
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        # 宽方向对半裁(本例为 0,不裁)
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        # center-crop: 切掉四周多补的边,得到干净的 721x1440 地面场
        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]
