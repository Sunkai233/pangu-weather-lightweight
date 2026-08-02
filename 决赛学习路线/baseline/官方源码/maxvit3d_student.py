# -*- coding: utf-8 -*-
"""
MaxViT-3D 学生模型  (盘古轻量化 · EXP-C1)
========================================================
设计要点:
  - 注意力 = 局部 block(连续窗口·抓细节) + 全局 grid(等距散布·抓大尺度环流),交替堆叠
  - QK-Norm: q,k 先 L2 归一化再点积(余弦注意力)+ 每头可学习温度(SwinV2 风格),训练更稳、可更深更窄
  - RMSNorm 替 LayerNorm; SwiGLU 门控 FFN; LayerScale + DropPath
  - I/O 与官方 Pangu 严格一致: 入 [B,72,721,1440] -> 出 (surface[B,4,721,1440], upper[B,5,13,721,1440])
  - stem(PatchEmbed2D/3D)与 head(PatchRecovery2D/3D)复用官方模块,保证形状与教师可蒸馏、可用官方 result.py 评测
  - 仅在中间“Fuser/骨干”用本文件的高表达力注意力替换官方 EarthAttention3D 窗口注意力
注: 当前为可运行脚手架(单分辨率骨干@embed_dim=192),后续可加 U 形多尺度 / 缩小 embed_dim。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from onescience.modules import OneEmbedding, OneRecovery, OneSample  # 工厂(SCNet/abc66 均有,style 名不同)


def _detect_pangu_api():
    """SCNet onescience 0.3.0 = 统一 PanguEmbedding;abc66(PyPI) = PanguEmbedding2D/3D 分版。"""
    try:
        OneEmbedding(style="PanguEmbedding", img_size=(721, 1440), patch_size=(4, 4), Variables=7, embed_dim=8)
        return "unified"
    except Exception:
        return "split"


_PANGU_API = _detect_pangu_api()


# ----------------------------- 基础组件 -----------------------------
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = x.new_empty(shape).bernoulli_(keep)
        if keep > 0:
            r.div_(keep)
        return x * r


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


class SwiGLU(nn.Module):
    """门控 FFN: (SiLU(xW1) ⊗ xW3) W2; 隐藏维取 ≈ 8/3·d 使参数与标准 4× GELU 相当。"""
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(mlp_ratio * dim * 2 / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w12 = nn.Linear(dim, hidden * 2)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(a) * b))


# ----------------------- 3D 分块: 局部 block / 全局 grid -----------------------
def _pad3d(x, win):
    """x:(B,Pl,Lat,Lon,C); 右侧补零到 win 整数倍。返回 (x_pad, (Pl,Lat,Lon)原始)。"""
    B, Pl, Lat, Lon, C = x.shape
    wz, wh, ww = win
    dz = (wz - Pl % wz) % wz
    dh = (wh - Lat % wh) % wh
    dw = (ww - Lon % ww) % ww
    if dz or dh or dw:
        x = F.pad(x, (0, 0, 0, dw, 0, dh, 0, dz))
    return x, (Pl, Lat, Lon)


def block_partition(x, win):
    """连续窗口=局部。 (B,Pl,Lat,Lon,C) -> (nW*B, wz*wh*ww, C)"""
    B, Pl, Lat, Lon, C = x.shape
    wz, wh, ww = win
    x = x.view(B, Pl // wz, wz, Lat // wh, wh, Lon // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wz * wh * ww, C)


def block_reverse(w, win, B, Pl, Lat, Lon):
    wz, wh, ww = win
    x = w.view(B, Pl // wz, Lat // wh, Lon // ww, wz, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, Pl, Lat, Lon, -1)


def grid_partition(x, grid):
    """跨步网格=全局。grid=(gz,gh,gw) 为每个注意力组内 token 数(空间等距散布)。
       (B,Pl,Lat,Lon,C) -> (nG*B, gz*gh*gw, C)"""
    B, Pl, Lat, Lon, C = x.shape
    gz, gh, gw = grid
    x = x.view(B, gz, Pl // gz, gh, Lat // gh, gw, Lon // gw, C)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(-1, gz * gh * gw, C)


def grid_reverse(w, grid, B, Pl, Lat, Lon):
    gz, gh, gw = grid
    x = w.view(B, Pl // gz, Lat // gh, Lon // gw, gz, gh, gw, -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(B, Pl, Lat, Lon, -1)


class RelPosBias3D(nn.Module):
    """3D 相对位置偏置(Swin 风格),用于窗口/网格内的 N×N 偏置。"""
    def __init__(self, win, num_heads):
        super().__init__()
        a, b, c = win
        self.win = win
        self.table = nn.Parameter(torch.zeros((2 * a - 1) * (2 * b - 1) * (2 * c - 1), num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(a), torch.arange(b), torch.arange(c), indexing="ij"))  # 3,a,b,c
        coords = torch.flatten(coords, 1)                       # 3, N
        rel = coords[:, :, None] - coords[:, None, :]           # 3, N, N
        rel = rel.permute(1, 2, 0).contiguous()                 # N, N, 3
        rel[:, :, 0] += a - 1
        rel[:, :, 1] += b - 1
        rel[:, :, 2] += c - 1
        rel[:, :, 0] *= (2 * b - 1) * (2 * c - 1)
        rel[:, :, 1] *= (2 * c - 1)
        idx = rel.sum(-1)                                       # N, N
        self.register_buffer("idx", idx, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self):
        a, b, c = self.win
        N = a * b * c
        bias = self.table[self.idx.view(-1)].view(N, N, -1)     # N,N,heads
        return bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # 1,heads,N,N


_USE_SDPA = False  # 置 True 用融合注意力(SDPA, 不物化 N×N, 省显存); maxvit3d_student._USE_SDPA = True


def set_sdpa(v: bool):
    global _USE_SDPA
    _USE_SDPA = bool(v)


class Attention3D(nn.Module):
    """带 QK-Norm 的多头注意力(余弦缩放 + 每头可学习温度)+ 相对位置偏置。"""
    def __init__(self, dim, win, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.bias = RelPosBias3D(win, num_heads)
        # QK-Norm 温度(每头),初始 log(10),上限 clamp(SwinV2)
        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))

    def forward(self, x):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # Bn,heads,N,hd
        q = F.normalize(q, dim=-1)                              # QK-Norm
        k = F.normalize(k, dim=-1)
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()  # heads,1,1
        if _USE_SDPA:
            # 融合注意力:不物化 (Bn,heads,N,N),省显存;τ 折进 q,bias 当 additive mask,内置 scale=1
            out = F.scaled_dot_product_attention(q * scale, k, v, attn_mask=self.bias(), scale=1.0)
            out = out.transpose(1, 2).reshape(Bn, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * scale           # Bn,heads,N,N
            attn = attn + self.bias()
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(out))


class SpectralMix2D(nn.Module):
    """轻量频域全局混合(AFNO 式):低模逐通道复滤波(depthwise)+ 通道线性混合。
    少参拿全局感受野(卷积定理 O(N log N));参数 = 2·C·mh·mw + C²(远小于 FNO 的 C²·modes)。"""
    def __init__(self, dim, modes=(12, 12)):
        super().__init__()
        self.mh, self.mw = modes
        self.fr = nn.Parameter(torch.ones(dim, self.mh, self.mw))    # 复滤波实部(初始~1=近恒等,稳)
        self.fi = nn.Parameter(torch.zeros(dim, self.mh, self.mw))   # 虚部初始 0
        self.proj = nn.Linear(dim, dim)                              # 通道混合

    def forward(self, h):  # h:(B,Pl,Lat,Lon,C)
        B, Pl, Lat, Lon, C = h.shape
        x = h.permute(0, 1, 4, 2, 3).reshape(B * Pl, C, Lat, Lon).float()
        ft = torch.fft.rfft2(x, norm="ortho")
        mh = min(self.mh, ft.shape[-2]); mw = min(self.mw, ft.shape[-1])
        filt = torch.complex(self.fr[:, :mh, :mw], self.fi[:, :mh, :mw])  # (C,mh,mw)
        out = torch.zeros_like(ft)
        out[:, :, :mh, :mw] = ft[:, :, :mh, :mw] * filt[None]            # depthwise 逐模缩放
        y = torch.fft.irfft2(out, s=(Lat, Lon), norm="ortho")
        y = y.reshape(B, Pl, C, Lat, Lon).permute(0, 1, 3, 4, 2)
        return self.proj(y)


class AttnFFN(nn.Module):
    """一个混合子层(block局部/grid全局/spectral频域全局)+ 一个 SwiGLU 子层, 均 pre-RMSNorm + LayerScale + DropPath。"""
    def __init__(self, dim, win, num_heads, mode, mlp_ratio=4.0, drop_path=0.0, ls_init=1e-5):
        super().__init__()
        assert mode in ("block", "grid", "spectral")
        self.mode = mode
        self.win = win
        self.norm1 = RMSNorm(dim)
        if mode == "spectral":
            self.mix = SpectralMix2D(dim)
        else:
            self.attn = Attention3D(dim, win, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.ls1 = nn.Parameter(ls_init * torch.ones(dim))
        self.ls2 = nn.Parameter(ls_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, res):
        Pl, Lat, Lon = res
        B, N, C = x.shape
        shortcut = x
        if self.mode == "spectral":
            h = self.norm1(x).view(B, Pl, Lat, Lon, C)
            h = self.mix(h).reshape(B, N, C)
        else:
            h = self.norm1(x).view(B, Pl, Lat, Lon, C)
            h, _ = _pad3d(h, self.win)
            pPl, pLat, pLon = h.shape[1:4]
            if self.mode == "block":
                w = block_partition(h, self.win)
                w = self.attn(w)
                h = block_reverse(w, self.win, B, pPl, pLat, pLon)
            else:
                w = grid_partition(h, self.win)
                w = self.attn(w)
                h = grid_reverse(w, self.win, B, pPl, pLat, pLon)
            h = h[:, :Pl, :Lat, :Lon, :].reshape(B, N, C)
        x = shortcut + self.drop_path(self.ls1 * h)
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x


# ----------------------------- 学生模型 -----------------------------
class MaxVit3DStudent(nn.Module):
    """U 形(encoder→downsample→mid→upsample→decoder)。
    大部分层在 1/4 token 的下采样分辨率上算 → 直取 V(延迟)/ U(显存)。
    depths=(enc, mid, dec) 为各阶段的 block+grid 对数; num_heads 为各阶段头数。"""
    def __init__(self, img_size=(721, 1440), patch_size=(2, 4, 4),
                 embed_dim=192, depths=(1, 3, 1), num_heads=(6, 12, 6),
                 window_size=(2, 6, 12), mlp_ratio=4.0, drop_path=0.1,
                 use_checkpoint=False, global_mode="grid"):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.global_mode = global_mode  # "grid"=稀疏全局注意力 / "spectral"=FNO频域全局
        # 复用官方 stem/head(embed_dim 可配 → 缩窄省显存/参数;I/O 与教师严格一致)。
        # 两套 onescience API 自适配:SCNet=统一 PanguEmbedding;abc66=PanguEmbedding2D/3D 分版。
        if _PANGU_API == "unified":
            self.patchembed2d = OneEmbedding(style="PanguEmbedding", img_size=img_size,
                                             patch_size=patch_size[1:], Variables=7, embed_dim=embed_dim)
            self.patchembed3d = OneEmbedding(style="PanguEmbedding", img_size=(13, *img_size),
                                             patch_size=patch_size, Variables=5, embed_dim=embed_dim)
            self.patchrecovery2d = OneRecovery(style="PanguPatchRecovery", img_size=img_size,
                                               patch_size=patch_size[1:], in_chans=2 * embed_dim, out_chans=4)
            self.patchrecovery3d = OneRecovery(style="PanguPatchRecovery", img_size=(13, *img_size),
                                               patch_size=patch_size, in_chans=2 * embed_dim, out_chans=5)
            self._ds_style, self._us_style = "PanguDownSample", "PanguUpSample"
        else:
            from onescience.modules.embedding.panguembedding2d import PanguEmbedding2D
            from onescience.modules.embedding.panguembedding3d import PanguEmbedding3D
            from onescience.modules.recovery.pangupatchrecovery2d import PanguPatchRecovery2D
            from onescience.modules.recovery.pangupatchrecovery3d import PanguPatchRecovery3D
            self.patchembed2d = PanguEmbedding2D(img_size=img_size, patch_size=patch_size[1:],
                                                 in_chans=7, embed_dim=embed_dim)
            self.patchembed3d = PanguEmbedding3D(img_size=(13, *img_size), patch_size=patch_size,
                                                 in_chans=5, embed_dim=embed_dim)
            self.patchrecovery2d = PanguPatchRecovery2D(img_size=img_size, patch_size=patch_size[1:],
                                                        in_chans=2 * embed_dim, out_chans=4)
            self.patchrecovery3d = PanguPatchRecovery3D(img_size=(13, *img_size), patch_size=patch_size,
                                                        in_chans=2 * embed_dim, out_chans=5)
            self._ds_style, self._us_style = "PanguDownSample3D", "PanguUpSample3D"

        Pl = 8
        Lat = math.ceil(img_size[0] / patch_size[1])   # 181
        Lon = math.ceil(img_size[1] / patch_size[2])   # 360
        Latd = math.ceil(Lat / 2)                       # 91
        Lond = math.ceil(Lon / 2)                       # 180
        self.res_full = (Pl, Lat, Lon)
        self.res_down = (Pl, Latd, Lond)

        # 复用官方下/上采样(I/O 与教师一致,权重从零随机初始化、随蒸馏训练)
        self.downsample = OneSample(style=self._ds_style, in_dim=embed_dim,
                                    input_resolution=self.res_full,
                                    output_resolution=self.res_down)
        self.upsample = OneSample(style=self._us_style, in_dim=embed_dim * 2,
                                  out_dim=embed_dim,
                                  input_resolution=self.res_down,
                                  output_resolution=self.res_full)

        ne, nm, nd = depths
        dpr = torch.linspace(0, drop_path, (ne + nm + nd) * 2).tolist()
        self._it = iter(dpr)

        def mk(n, dim, heads):
            blks = []
            for _ in range(n):
                blks.append(AttnFFN(dim, window_size, heads, "block", mlp_ratio, next(self._it)))
                blks.append(AttnFFN(dim, window_size, heads, self.global_mode, mlp_ratio, next(self._it)))
            return nn.ModuleList(blks)

        self.enc = mk(ne, embed_dim, num_heads[0])
        self.mid = mk(nm, embed_dim * 2, num_heads[1])
        self.dec = mk(nd, embed_dim, num_heads[2])

    def _run(self, blocks, x, res):
        for blk in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, res, use_reentrant=False)
            else:
                x = blk(x, res)
        return x

    def forward(self, x):
        surface = x[:, :7, :, :]
        upper = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)                       # B,192,181,360
        upper = self.patchembed3d(upper)                           # B,192,7,181,360
        x = torch.concat([surface.unsqueeze(2), upper], dim=2)     # B,192,8,181,360
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)                    # B,N,192

        x = self._run(self.enc, x, self.res_full)                  # 全分辨率
        skip = x
        x = self.downsample(x)                                     # B,N',384
        x = self._run(self.mid, x, self.res_down)                  # 1/4 token(算力大头在此)
        x = self.upsample(x)                                       # B,N,192
        x = self._run(self.dec, x, self.res_full)

        output = torch.concat([x, skip], dim=-1)                   # B,N,384
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        out_surface = self.patchrecovery2d(output[:, :, 0, :, :])  # B,4,721,1440
        out_upper = self.patchrecovery3d(output[:, :, 1:, :, :])   # B,5,13,721,1440
        return out_surface, out_upper


if __name__ == "__main__":
    m = MaxVit3DStudent()
    n = sum(p.numel() for p in m.parameters())
    print(f"MaxVit3DStudent params: {n/1e6:.2f} M")
