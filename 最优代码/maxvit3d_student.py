# -*- coding: utf-8 -*-
"""
MaxViT-3D 学生模型  (盘古轻量化 · EXP-C1)
========================================================
设计要点:
  - 注意力 = 局部 block(连续窗口·抓细节) + 全局支路,交替堆叠
      全局支路三选一(global_mode): grid(等距散布稀疏) / spectral(FNO 频域) / fpa(原型自适应:
      原型按真实嵌入分布自适应 + token 查询匹配,内容自适应、O(N·M)、DCU 稠密 GEMM 友好)
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

# ★推理期注意力分块数(显存优化,纯 PyTorch,无需编译):
#   注意力分数矩阵 (Bn, heads, N, N) 中 Bn=窗口数,各窗口 softmax 相互独立,
#   按 Bn 维分成 _ATTN_CHUNKS 块顺序计算,与全量在数学上完全等价,
#   但峰值显存从整块降到 1/_ATTN_CHUNKS。1=关闭(全量);推荐 8。
import os as _os
_ATTN_CHUNKS = int(_os.environ.get("ATTN_CHUNKS", "8"))


def set_attn_chunks(n: int):
    global _ATTN_CHUNKS
    _ATTN_CHUNKS = max(1, int(n))


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
        elif not self.training and _ATTN_CHUNKS > 1 and Bn > _ATTN_CHUNKS:
            # ★显存优化(推理期):按窗口(Bn)维分块计算注意力。
            #   各窗口的 softmax 本就相互独立,分块与全量在数学上完全等价(逐行 softmax 不跨窗口),
            #   但分数矩阵峰值从 (Bn,heads,N,N) 降到 (chunk,heads,N,N),显存峰值降约 _ATTN_CHUNKS 倍。
            #   纯 PyTorch 实现,无需任何编译/自定义算子;与全量前向数值一致(仅浮点累加顺序不变)。
            bias = self.bias()
            step = (Bn + _ATTN_CHUNKS - 1) // _ATTN_CHUNKS
            out = torch.empty(Bn, N, C, dtype=x.dtype, device=x.device)
            for i in range(0, Bn, step):
                j = min(i + step, Bn)
                a = (q[i:j] @ k[i:j].transpose(-2, -1)) * scale       # (chunk,heads,N,N)
                a = a + bias
                a = a.softmax(dim=-1)
                out[i:j] = (a @ v[i:j]).transpose(1, 2).reshape(j - i, N, C)
                del a
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


class PrototypeAdaptiveMix(nn.Module):
    """原型自适应全局混合(FPA, Feature-Prototype Adaptive)。

    把 grid 的「定步等距、内容无关」全局采样,换成「自适应贴合真实嵌入分布」的原型集:
      ▸ Phase-1 原型自适应(可微聚类 / 竞争式软分配, slot-attention 风格):
        M 个原型种子(可学习初始)经 T 次迭代被本输入的真实 token 嵌入「竞争式吸收」,
        收敛到该输入嵌入分布的质心 —— 自动迁移到急流 / 阻塞 / 气旋等动力学活跃区
        (grid 的固定格点做不到)。softmax 在「原型维(M)」归一 → 原型相互竞争、铺开覆盖
        分布而不塌缩;再在「token 维(N)」归一得加权均值 = 质心。残差 + RMSNorm 更新稳收敛。
      ▸ Phase-2 查询匹配(token ← 原型):
        每个 token 以自身为 query,对 M 个原型(K/V)做交叉注意力取全局上下文。
        复杂度 O(N·M)(M≪N),且全程稠密 GEMM + softmax → rocBLAS/flash 友好、无 partition 拷贝。
    两个 phase 都带 QK-Norm(余弦相似 + 每头可学习温度)稳训,与项目其它注意力一致。
    输入 x 已由外层 AttnFFN 做过 pre-RMSNorm。
    """
    def __init__(self, dim, num_heads, num_prototypes=128, n_iters=2, qkv_bias=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.hd = dim // num_heads
        self.M = num_prototypes
        self.T = n_iters
        self.eps = eps
        # 原型种子:可学习初始,前向时按本输入真实嵌入分布自适应
        self.proto = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        # Phase-1 投影(原型→q、token→k/v)+ 残差更新
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.upd = nn.Linear(dim, dim)
        self.norm_p = RMSNorm(dim)
        self.scale1 = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))
        # Phase-2 投影(token→q、原型→k/v)+ 输出
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale2 = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))
        self.proj = nn.Linear(dim, dim)

    def _heads(self, t, B, L):          # (B,L,C) -> (B,h,L,hd)
        return t.view(B, L, self.h, self.hd).permute(0, 2, 1, 3)

    def forward(self, x):               # x:(B,N,C),已 pre-norm
        B, N, C = x.shape
        K1 = F.normalize(self._heads(self.k1(x), B, N), dim=-1)      # B,h,N,hd
        V1 = self._heads(self.v1(x), B, N)
        P = self.proto.unsqueeze(0).expand(B, -1, -1).contiguous()  # B,M,C
        t1 = torch.clamp(self.scale1, max=math.log(100.0)).exp()
        # Phase-1: 原型竞争式吸收真实嵌入分布(可微聚类)
        for _ in range(self.T):
            Q1 = F.normalize(self._heads(self.q1(P), B, self.M), dim=-1)   # B,h,M,hd
            a = (Q1 @ K1.transpose(-2, -1)) * t1                          # B,h,M,N
            a = a.softmax(dim=-2)                                         # 原型维竞争(slot)
            a = a / (a.sum(dim=-1, keepdim=True) + self.eps)             # token 维归一→质心
            upd = (a @ V1).permute(0, 2, 1, 3).reshape(B, self.M, C)     # B,M,C
            P = self.norm_p(P + self.upd(upd))                           # 残差更新
        # Phase-2: token 查询匹配原型,取全局上下文 O(N·M)
        Q2 = F.normalize(self._heads(self.q2(x), B, N), dim=-1)         # B,h,N,hd
        K2 = F.normalize(self._heads(self.k2(P), B, self.M), dim=-1)    # B,h,M,hd
        V2 = self._heads(self.v2(P), B, self.M)
        t2 = torch.clamp(self.scale2, max=math.log(100.0)).exp()
        g = ((Q2 @ K2.transpose(-2, -1)) * t2).softmax(dim=-1) @ V2     # B,h,N,hd
        g = g.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(g)


class AttnFFN(nn.Module):
    """一个混合子层(block局部 / grid全局 / spectral频域全局 / fpa原型自适应全局)+ 一个 SwiGLU 子层,
    均 pre-RMSNorm + LayerScale + DropPath。"""
    def __init__(self, dim, win, num_heads, mode, mlp_ratio=4.0, drop_path=0.0, ls_init=1e-5):
        super().__init__()
        assert mode in ("block", "grid", "spectral", "fpa")
        self.mode = mode
        self.win = win
        self.norm1 = RMSNorm(dim)
        if mode == "spectral":
            self.mix = SpectralMix2D(dim)
        elif mode == "fpa":
            self.mix = PrototypeAdaptiveMix(dim, num_heads)
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
        elif self.mode == "fpa":
            h = self.mix(self.norm1(x))          # 直接在 (B,N,C) 上,无需空间分块/重排
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
        self.global_mode = global_mode  # "grid"=定步稀疏全局 / "spectral"=FNO频域全局 / "fpa"=原型自适应全局
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

    # ---------------- 显存优化辅助(推理用,数学等价) ----------------
    @staticmethod
    def _embed_tiled(embed_mod, x5, n_tiles):
        """阶段0 conv快赢:patch conv(stride==kernel 非重叠)→ 等价 unfold(im2col)+linear(GEMM)。
        每个非重叠 patch 独立投影 = 数学等价的 Conv,但走 hipBLAS GEMM,不触发 MIOpen conv kernel
        的 JIT 搜索/编译(评测首样本 conv 编译 ~5s 大头)。matmul 用 fp32 累加保最大保真(fp16 误差
        ~4e-3,与 MIOpen conv 自身 fp16 误差同量级、均远小于 1e-2 容差,W 不掉点)。
        仍沿纬度(token-H)分块以保持显存峰值(U)优化。"""
        inner = list(embed_mod.children())[0]          # PanguEmbedding
        pad, proj = inner.pad, inner.proj
        sq = False
        if x5.ndim == 4:
            x5 = x5.unsqueeze(2); sq = True
        W = proj.weight                                # (O, V, kz, kh, kw)
        b = proj.bias                                  # (O,)
        O, V, kz, kh, kw = W.shape
        Wf = W.reshape(O, V * kz * kh * kw).t().float()   # (K, O), fp32
        bf = b.float()
        xp = pad(x5)
        _, _, Plp, Hp, Wp = xp.shape
        nz, nH, nw = Plp // kz, Hp // kh, Wp // kw
        rows = math.ceil(nH / n_tiles)
        outs = []
        for s in range(0, nH, rows):
            e = min(s + rows, nH)
            blk = xp[:, :, :, s * kh:e * kh, :]            # B,V,Plp,h*kh,Wp
            Bb = blk.shape[0]; nh = e - s
            blk = blk.reshape(Bb, V, nz, kz, nh, kh, nw, kw)
            blk = blk.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # B,nz,nh,nw,V,kz,kh,kw
            blk = blk.reshape(Bb, nz * nh * nw, V * kz * kh * kw)    # B,Ntok,K
            o = (torch.matmul(blk.float(), Wf) + bf).to(W.dtype)     # B,Ntok,O
            o = o.reshape(Bb, nz, nh, nw, O).permute(0, 4, 1, 2, 3).contiguous()  # B,O,nz,nh,nw
            outs.append(o); del blk, o
        out = torch.cat(outs, dim=3); del xp, outs
        return out.squeeze(2) if sq else out

    @staticmethod
    def _recovery_tiled(rec_mod, tok, n_tiles):
        """阶段0 conv快赢:patch convT(stride==kernel 非重叠)→ 等价 linear(GEMM)+fold+对称裁剪。
        每个 token 独立生成一个 (out_chans,kz,kh,kw) 块 = 数学等价的 ConvTranspose,但走 hipBLAS GEMM,
        不触发 MIOpen 反卷积 kernel JIT 编译。实测此路径 fp16 与原 convT bit-exact(max diff=0)。
        仍沿 token-H 分块以保持显存峰值(U)优化。"""
        inner = list(rec_mod.children())[0]            # PanguPatchRecovery
        proj, img = inner.proj, inner.img_size
        sq = False
        if tok.ndim == 4:
            tok = tok.unsqueeze(2); sq = True
        W = proj.weight                                # (Cin, O, kz, kh, kw)
        b = proj.bias                                  # (O,)
        Cin, O, kz, kh, kw = W.shape
        Wf = W.reshape(Cin, O * kz * kh * kw)          # (Cin, O*kz*kh*kw)
        _, _, Plt, H_, Wt = tok.shape
        rows = math.ceil(H_ / n_tiles)
        outs = []
        for s in range(0, H_, rows):
            e = min(s + rows, H_)
            t = tok[:, :, :, s:e, :]                    # B,Cin,Plt,nh,Wt
            Bb = t.shape[0]; nh = e - s
            t = t.permute(0, 2, 3, 4, 1).contiguous().reshape(Bb, Plt * nh * Wt, Cin)  # B,N,Cin
            o = torch.matmul(t, Wf)                     # B,N,O*kz*kh*kw  (bias 后加)
            o = o.reshape(Bb, Plt, nh, Wt, O, kz, kh, kw)
            o = o.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()  # B,O,Plt,kz,nh,kh,Wt,kw
            o = o.reshape(Bb, O, Plt * kz, nh * kh, Wt * kw)
            outs.append(o); del t, o
        output = torch.cat(outs, dim=3); del outs
        output = output + b.view(1, O, 1, 1, 1)         # bias(与 convT 一致,逐输出通道)
        _, _, PL, H, W2 = output.shape
        PLp, Hp, Wp = PL - img[0], H - img[1], W2 - img[2]
        pf, pt, pl = PLp // 2, Hp // 2, Wp // 2
        output = output[:, :, pf:PL - (PLp - pf), pt:H - (Hp - pt), pl:W2 - (Wp - pl)]
        return output.squeeze(2) if sq else output

    def _block_memopt(self, blk, x, res):
        """复刻 AttnFFN.forward,但中间张量用完即 del(数学等价)。窗口注意力天然按窗口独立。"""
        Pl, Lat, Lon = res
        B, N, C = x.shape
        shortcut = x
        if blk.mode == "spectral":
            h = blk.norm1(x).view(B, Pl, Lat, Lon, C)
            h = blk.mix(h).reshape(B, N, C)
        elif blk.mode == "fpa":
            h = blk.mix(blk.norm1(x))
        else:
            h = blk.norm1(x).view(B, Pl, Lat, Lon, C)
            h, _ = _pad3d(h, blk.win)
            pPl, pLat, pLon = h.shape[1:4]
            if blk.mode == "block":
                w = block_partition(h, blk.win); w = blk.attn(w)
                h = block_reverse(w, blk.win, B, pPl, pLat, pLon)
            else:
                w = grid_partition(h, blk.win); w = blk.attn(w)
                h = grid_reverse(w, blk.win, B, pPl, pLat, pLon)
            del w
            h = h[:, :Pl, :Lat, :Lon, :].reshape(B, N, C)
        x = shortcut + blk.drop_path(blk.ls1 * h); del h, shortcut
        x = x + blk.drop_path(blk.ls2 * blk.mlp(blk.norm2(x)))
        return x

    def _run_memopt(self, blocks, x, res):
        for blk in blocks:
            x = self._block_memopt(blk, x, res)
        return x

    def forward(self, x):
        # mem_opt: 推理期显存优化路径(数学等价,实测峰值 923MB->548MB,W/RMSE 不变)。
        #   ① stem/head 高分辨率 conv 沿纬度分块(stride=kernel 非重叠 → bit-exact)
        #   ② stem 后立即释放 fp32/输入张量存储(贯穿全程的最大常驻项)
        #   ③ encoder/decoder 逐 block 中间张量用完即 del
        # 注意:推荐配合 set_sdpa(False)(本 DCU 的 SDPA 回退工作集巨大,naive 注意力反而省 ~200MB)。
        mem_opt = getattr(self, "mem_opt", True) and not self.training
        if not mem_opt:
            return self._forward_plain(x)
        tile = getattr(self, "mem_tile", 6)
        surface = x[:, :7, :, :]
        upper = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self._embed_tiled(self.patchembed2d, surface, tile)
        upper = self._embed_tiled(self.patchembed3d, upper, tile)
        x.untyped_storage().resize_(0)                  # 释放输入(caller 此后不再用 invar)
        xx = torch.concat([surface.unsqueeze(2), upper], dim=2)
        del surface, upper
        B, C, Pl, Lat, Lon = xx.shape
        xx = xx.reshape(B, C, -1).transpose(1, 2).contiguous()
        xx = self._run_memopt(self.enc, xx, self.res_full)
        skip = xx
        xx = self.downsample(xx)
        xx = self._run_memopt(self.mid, xx, self.res_down)
        xx = self.upsample(xx)
        xx = self._run_memopt(self.dec, xx, self.res_full)
        output = torch.concat([xx, skip], dim=-1)
        del xx, skip
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        rin2 = output[:, :, 0, :, :].contiguous()
        rin3 = output[:, :, 1:, :, :].contiguous()
        del output
        out_surface = self._recovery_tiled(self.patchrecovery2d, rin2, tile); del rin2
        out_upper = self._recovery_tiled(self.patchrecovery3d, rin3, tile); del rin3
        return out_surface, out_upper

    def _forward_plain(self, x):
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
