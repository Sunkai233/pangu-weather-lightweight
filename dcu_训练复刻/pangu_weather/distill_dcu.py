# -*- coding: utf-8 -*-
"""
教师-学生蒸馏训练 —— 海光 DCU 版(64GB 内存约束下的完整可复现训练)
================================================================================
教师 : 官方 Pangu 全量权重(data/checkpoints/model_bak.pth),冻结,只做前向
学生 : MaxVit3DStudent(自研结构,权重完全由蒸馏得到 —— 不存在从零训练)
数据 : 官方 ERA5Datapipe(共享数据集 old-data)
损失 : alpha·L1(学生, 教师输出) + (1-alpha)·L1(学生, ERA5 真值)  —— 双监督
       变量权重沿用官方 train.py 口径(surface 0.25 / upper 1.0,逐通道 weights)

内存与速度的取舍(本文件相对原 distill_scnet.py 的主要改动)
--------------------------------------------------------------------------------
教师 Pangu 是 64.22M 的全量模型,一次前向约 0.5s,比学生的前向+反向(约 75ms)贵 7 倍。
若每个 epoch 都重跑教师,训练时间的 ~87% 都花在教师上。
但教师对同一样本的输出是常量,只需算一次 —— 于是缓存。

  · 直接 fp16 缓存:每样本 69ch×721×1440×2B = 143MB。DCU 容器只有 64GB 内存,
    刨去系统与数据缓冲,可用约 40GB → 只装得下 ~280 个样本,训练集被迫大幅缩水。
  · 落磁盘缓存:143MB 从共享存储读回来约 0.5s —— 和重跑一遍教师一样慢,没有意义。
  · ★本文件采用 int8 逐通道量化缓存:按通道取 |max| 定标,存储减半到 71MB/样本,
    同样 40GB 可装 ~560 个样本。反量化开销约 5ms,相对省下的 500ms 教师前向可忽略。
    量化相对误差约 0.4%,而蒸馏用的是 L1 损失、教师输出自身也带误差,该噪声可忽略。

缓存放 CPU 内存(不占 DCU 显存),按 --cache-gb 设上限,装满即停:未命中的样本回退为
实时跑教师,因此任何内存预算下都能跑通,只是加速比随命中率变化。

断点续训:容器有 4 小时上限,每个 epoch 结束都会写 .ckpt(权重+优化器+epoch),
用 --resume 可无损接着跑,不会从头开始。

复现最优权重(student.pth, 5.19M, cfg={embed96, depths[2,4,2], heads[6,12,6],
patch[2,16,16], grid, mlp_ratio 2.0}):
  python distill_dcu.py --embed 96 --depths 2,4,2 --heads 6,12,6 --patch 2,16,16 \
      --global-mode grid --mlp-ratio 2.0 --alpha 0.5 --lr 6e-4 --epochs 60 \
      --cache-gb 40 --save student.pth
================================================================================
"""
import os, sys, time, math, argparse
import numpy as np, torch
import torch.nn.functional as F
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
import maxvit3d_student as M
from maxvit3d_student import MaxVit3DStudent


def wl1(x, y, w, lw):
    """官方 train.py 的加权 L1:逐通道权重 w,层级权重 lw。"""
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


def _newtonschulz5(G, steps=6, eps=1e-7):
    """Newton-Schulz 五次迭代,把梯度矩阵近似正交化(求 G 的"零次幂" ≈ UV^T)。
    系数 (3.4445, -4.7750, 2.0315) 取自 Keller Jordan 的 Muon 官方实现。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon 优化器(内置实现 —— 评测容器无外网,装不了 muon 包)。

    思路:对【骨干里的 2D 权重】,先做 Nesterov 动量,再用 Newton-Schulz 迭代把动量矩阵
    正交化后再更新 —— 相当于沿"各方向等步长"的方向走,比 AdamW 的逐元素自适应更适合
    矩阵型权重。历史实验中 momentum=0.99(而非默认 0.95)是最优项。
    patch embedding / recovery / logit_scale 不走 Muon,留给 AdamW。
    """
    def __init__(self, params, lr=0.02, momentum=0.99, nesterov=True, ns_steps=6):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, mom, nest, ns = g["lr"], g["momentum"], g["nesterov"], g["ns_steps"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(grad)
                buf = st["mom"]
                buf.mul_(mom).add_(grad)
                d = grad.add(buf, alpha=mom) if nest else buf
                d = _newtonschulz5(d, steps=ns).to(p.dtype)
                # 按形状缩放,使不同尺寸的权重更新幅度可比
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(d, alpha=-lr * scale)


def spectral_loss(pred, target, lw):
    """频域蒸馏(平面 FFT):对齐师生输出的 2D 功率谱(log-power L1)。
    学生倾向输出过平滑的场 → 高频功率偏低 → log-power 之差被惩罚 → 逼学生把高频细节补回来。
    FFT 对 fp16 不稳,故在 autocast 之外用 float 计算。"""
    pf = torch.fft.rfft2(pred.float(), norm="ortho")
    tf = torch.fft.rfft2(target.float(), norm="ortho")
    lp = torch.log(pf.real ** 2 + pf.imag ** 2 + 1e-6)
    lt = torch.log(tf.real ** 2 + tf.imag ** 2 + 1e-6)
    return lw * (lp - lt).abs().mean()


_SHT = None


def spherical_spectral_loss(pred, target, lw):
    """球面谱蒸馏(球谐 SHT):对齐师生的角功率谱 C_ℓ = Σ_m |a_ℓm|²。
    ERA5 是球面场,平面 FFT 在极区(经线收敛处)会失真;球谐变换几何正确,
    C_ℓ 按球谐阶 ℓ 天然分尺度。学生过平滑 → 高 ℓ(小尺度)功率偏低 → 被罚。"""
    global _SHT
    if _SHT is None:
        from torch_harmonics import RealSHT
        H, W = pred.shape[-2], pred.shape[-1]
        _SHT = RealSHT(H, W, grid="equiangular").to(pred.device)
    pf = _SHT(pred.float()); tf = _SHT(target.float())
    pc = (pf.real ** 2 + pf.imag ** 2).sum(dim=-1)
    tc = (tf.real ** 2 + tf.imag ** 2).sum(dim=-1)
    return lw * (torch.log(pc + 1e-8) - torch.log(tc + 1e-8)).abs().mean()


@torch.no_grad()
def ema_update(ema, model, d=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


class SampleCache:
    """整样本的 int8 逐通道量化缓存(常驻 CPU 内存)。

    为什么要连输入和真值一起缓存,而不是只缓存教师输出:
      每个样本的原始数据是一个 393MB 的 h5 文件,从共享存储(NFS)读回来约 2.3 秒,
      而教师前向只要约 0.5 秒 —— 实测每样本 2.8 秒里,【IO 才是大头】。
      若只缓存教师输出,第二轮起虽免了教师,却仍要重新读盘,每个 epoch 依旧 2 小时以上,
      60 个 epoch 根本跑不完。把输入 x、真值、教师输出一并缓存,第二轮起彻底不碰磁盘,
      每个 epoch 只剩学生的前向反向(约 80 秒),整个训练缩到 2 小时以内。

    量化:这些都是归一化后的气象场,逐通道用 |max|/127 定标即可,不需要 zero-point。
      int8 让每样本从 fp16 的 435MB 降到约 216MB,同样内存能多缓存一倍的样本。
      量化相对误差约 0.4%,而蒸馏用的是 L1 损失、教师输出自身也带误差,该噪声可忽略。

    缓存放 CPU 内存(不占 DCU 显存),按预算装满即停 —— 未命中的样本回退为读盘+跑教师,
    因此任何内存预算下都能跑通,只是加速比随命中率变化。
    """
    def __init__(self, budget_gb):
        self.budget = int(float(budget_gb) * (1 << 30))
        self.used = 0
        self.store = {}
        self.hit = 0
        self.miss = 0

    @staticmethod
    def _q(t):
        s = t.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / 127.0
        return (t / s).round().clamp_(-127, 127).to(torch.int8).cpu(), s.float().cpu()

    @staticmethod
    def _dq(qs, dev):
        q, s = qs
        return q.to(dev, non_blocking=True).float() * s.to(dev, non_blocking=True)

    def get(self, key, dev):
        if key is None:
            return None
        e = self.store.get(key)
        if e is None:
            self.miss += 1
            return None
        self.hit += 1
        return tuple(self._dq(z, dev) for z in e)     # x, tsf, tua, ts, tu

    def put(self, key, x, tsf, tua, ts, tu):
        if key is None or key in self.store:
            return
        packs = [self._q(t.detach().float()) for t in (x, tsf, tua, ts, tu)]
        n = sum(q.numel() + 4 * s.numel() for q, s in packs)
        if self.used + n > self.budget:
            return                                    # 预算用尽:此样本每轮实时读盘+跑教师
        self.store[key] = tuple(packs)
        self.used += n

    def stat(self):
        tot = self.hit + self.miss
        return (len(self.store), self.used / (1 << 30),
                100.0 * self.hit / tot if tot else 0.0)


def sample_key(data):
    """ERA5Datapipe 每个样本自带文件名(inference.py 里用 data[4][-1][0] 命名输出),
    用它做缓存键,即使 dataloader 打乱顺序也能稳定命中。"""
    try:
        return str(data[4][-1][0])
    except Exception:
        return None


class CachedDataset(torch.utils.data.Dataset):
    """从 prep_cache.py 预处理好的 int8 缓存里流式读样本。

    这是绕开两堵墙的正解:
      ① 容器内存墙(cgroup 只有 59GB,装不下几百个样本)——内存里一个样本都不常驻
      ② h5py 解析墙(裸读 411MB 的 h5 只要 0.14s,但 h5py 从 99 个通道里挑 69 个要 4.5s)
         ——教师前向与 h5 解析都只在预处理阶段做一次,永久存下来

    每样本 216MB(int8),从盘里读回只要 0.05s(4.4 GB/s);配 4 个 worker 预取,
    读盘完全被 GPU 计算(0.24s)掩盖。且【每个 epoch 随机遍历全部样本】——
    标准 SGD,梯度无偏、不存在灾难性遗忘(这正是"池子轮换"方案的软肋)。
    """
    def __init__(self, root):
        self.files = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz"))
        if not self.files:
            raise RuntimeError(f"缓存目录为空: {root}(先跑 prep_cache.py)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        out = []
        for k, ks in (("x", "xs"), ("tsf", "tsfs"), ("tua", "tuas"), ("ts", "tss"), ("tu", "tus")):
            q = torch.from_numpy(z[k])                       # int8
            s = torch.from_numpy(z[ks])                      # 逐通道 scale
            out.append((q, s))
        return out                                            # 反量化留到 GPU 上做(省 CPU 和带宽)


def dequant_batch(item, dev):
    """[(int8, scale)] × 5 → GPU 上的 fp32 张量。反量化放 GPU:CPU 只搬 int8,带宽减半。"""
    res = []
    for q, s in item:
        q = q.to(dev, non_blocking=True).float()
        s = s.to(dev, non_blocking=True)
        res.append((q * s).squeeze(0) if q.dim() == 5 else q * s)
    return res


def main():
    ap = argparse.ArgumentParser()
    # ── 学生结构(必须与最优权重的 cfg 一致才能复现)──
    ap.add_argument("--embed", type=int, default=96)
    ap.add_argument("--depths", type=str, default="2,4,2")
    ap.add_argument("--heads", type=str, default="6,12,6")
    ap.add_argument("--patch", type=str, default="2,16,16")
    ap.add_argument("--global-mode", dest="gm", type=str, default="grid", choices=["grid", "block"])
    ap.add_argument("--mlp-ratio", dest="mlp_ratio", type=float, default=2.0,
                    help="SwiGLU FFN 宽度比。最优权重用 2.0,原 distill_scnet.py 没有这个开关且"
                         "保存的 cfg 里也缺它,推理端会退回默认 4.0 → 结构对不上,必须显式写入")
    ap.add_argument("--sdpa", type=int, default=1)
    # ── 训练 ──
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.5, help="蒸馏权重;(1-alpha) 为对真值的权重")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--opt", type=str, default="adamw", choices=["adamw", "muon"],
                    help="★历史最优用 muon:2D 权重走 Muon、其余走 AdamW")
    ap.add_argument("--muon-lr", dest="muon_lr", type=float, default=0.02,
                    help="Muon 学习率(比 AdamW 大一个量级)")
    ap.add_argument("--muon-momentum", dest="muon_mom", type=float, default=0.99,
                    help="Muon 动量。★历史最优为 0.99(默认 0.95 更抖)")
    ap.add_argument("--spec", type=float, default=0.0,
                    help="频域蒸馏系数(治学生输出过平滑),0=关闭。历史实验用 0.1")
    ap.add_argument("--spec-mode", dest="spec_mode", type=str, default="plane",
                    choices=["plane", "sphere"],
                    help="谱损失模式:plane=平面FFT / sphere=球谐SHT(球面几何正确,需 torch_harmonics)")
    ap.add_argument("--aug", type=int, default=1,
                    help="经度增广:沿经度做周期性 roll。地球经度本就是周期的,这是免费的数据增强")
    ap.add_argument("--grad-ckpt", dest="gckpt", type=int, default=1, help="梯度检查点(省显存)")
    # ── 内存 / 断点 ──
    ap.add_argument("--cache-gb", dest="cache_gb", type=float, default=38.0,
                    help="内存池上限(GB)。★注意容器的真实内存要读 cgroup,不能信 free —— "
                         "free 报的是宿主机(503GB),而 /sys/fs/cgroup/memory/memory.limit_in_bytes "
                         "只有 59GB。给缓存 38GB(约 180 个样本),其余留给模型/worker/系统。"
                         "设 0 则完全不缓存(最省内存,但每轮都要读盘+跑教师,慢约 25 倍)")
    ap.add_argument("--pool-size", dest="pool", type=int, default=180,
                    help="内存池能同时装下的样本数(受 --cache-gb 限制,约 cache_gb*1024/216)")
    ap.add_argument("--epochs-per-pool", dest="epp", type=int, default=12,
                    help="每个池子训多少轮再换下一批样本。★这是绕开内存墙的关键:"
                         "内存只装得下 180 个样本,但训练【一共见过多少样本】不受内存限制 —— "
                         "灌一池训 N 轮、清空换下一批,总样本数 = (epochs/epp) * pool_size。"
                         "默认 60/12=5 个池子 × 180 = 见过 900 个样本。调小此值 → 样本更多但灌池更频繁")
    ap.add_argument("--max-iters", type=int, default=0,
                    help=">0 限制每轮迭代数(冒烟用)。正式训练用 --pool-size 控制,不要用这个")
    ap.add_argument("--max-val", type=int, default=32,
                    help="每个 epoch 验证多少个样本。验证要对每个样本跑一次教师(约 1s),"
                         "而一整年就有 1459 个样本 —— 全量验证一次要 20 多分钟,比训练还贵。"
                         "取固定的前 N 个样本即可稳定反映趋势;设 0 表示全量验证")
    ap.add_argument("--resume", type=str, default="", help="从 .ckpt 续训(容器 4 小时上限用)")
    ap.add_argument("--save", type=str, default="student.pth")
    ap.add_argument("--config", type=str, default="conf/config_train.yaml",
                    help="训练用配置(指向共享 ERA5 数据集)。切勿改用 conf/config.yaml —— "
                         "那是推理/评测配置(test_ratio 为 2050~2058),改动会导致评测失败")
    ap.add_argument("--cache-dir", dest="cdir", type=str, default="",
                    help="★推荐路径:指向 prep_cache.py 预处理好的 int8 缓存目录。"
                         "此时不再读 h5、不再跑教师(教师输出已存盘),每个 epoch 随机遍历全部样本 —— "
                         "标准 SGD,梯度无偏、无遗忘,且内存里一个样本都不常驻(绕开 59GB 的 cgroup 墙)。"
                         "留空则回退到直读 h5 的慢路径(每样本 5.5s,其中 4.5s 耗在 h5py 解析上)")
    ap.add_argument("--loader-workers", dest="lw", type=int, default=4,
                    help="从缓存流式读的 worker 数(读 216MB 只要 0.05s,4 个足以被 GPU 计算掩盖)")
    args = ap.parse_args()

    depths = tuple(int(v) for v in args.depths.split(","))
    heads = tuple(int(v) for v in args.heads.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))
    dev = 0
    torch.backends.cuda.matmul.allow_tf32 = True
    M.set_sdpa(bool(args.sdpa))

    cur = os.getcwd(); sys.path.append(cur)
    cfgp = args.config if os.path.isabs(args.config) else os.path.join(cur, args.config)
    if not os.path.exists(cfgp):
        raise FileNotFoundError(f"训练配置不存在: {cfgp}(见 make_config_train.py 生成)")
    cfg = YParams(cfgp, "model")
    cfg_data = YParams(cfgp, "datapipe")
    print(f"[config] {cfgp}", flush=True)
    print(f"[data]   dir={cfg_data.dataset.data_dir}", flush=True)
    print(f"[split]  train={list(cfg_data.dataset.train_ratio)} val={list(cfg_data.dataset.val_ratio)}", flush=True)

    FAST = bool(args.cdir) and os.path.isdir(os.path.join(args.cdir, "train"))

    sw = torch.as_tensor(cfg_data.dataset.weights[:4], device=dev, dtype=torch.float32).view(1, -1, 1, 1)
    pw = torch.as_tensor(cfg_data.dataset.weights[4:], device=dev, dtype=torch.float32).view(1, -1, 1, 1)

    if FAST:
        # ★ 快速路径:从 prep_cache.py 预处理好的 int8 缓存流式读。
        #   教师输出与 h5 解析都已在预处理阶段做完并存盘,这里既不读 h5、也不加载教师 —— 省 1.8GB 显存。
        #   每个 epoch 随机遍历全部样本(shuffle=True),标准 SGD:梯度无偏、无灾难性遗忘。
        tr_ds = CachedDataset(os.path.join(args.cdir, "train"))
        va_ds = CachedDataset(os.path.join(args.cdir, "val"))
        train_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=None, shuffle=True, num_workers=args.lw,
            persistent_workers=args.lw > 0, prefetch_factor=2 if args.lw > 0 else None)
        val_loader = torch.utils.data.DataLoader(
            va_ds, batch_size=None, shuffle=False, num_workers=min(2, args.lw),
            persistent_workers=args.lw > 0)
        teacher = None
        tp = 64.22e6                                  # 官方 Pangu 全量参数量(仅用于打印压缩比)
        print(f"[cache] 预处理缓存: train {len(tr_ds)} 个 / val {len(va_ds)} 个 @ {args.cdir}", flush=True)
        print(f"[cache] 不读 h5、不跑教师(教师输出已存盘);每样本读盘约 0.05s,"
              f"{args.lw} 个 worker 预取即可被 GPU 计算完全掩盖", flush=True)
        surface_mask = None
    else:
        # 慢路径(兼容):直读 h5 + 实时跑教师。每样本约 5.5s,其中 4.5s 耗在 h5py 的通道选择上。
        datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
        train_loader, _ = datapipe.train_dataloader()
        val_loader, _ = datapipe.val_dataloader()
        print(f"[data] train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)
        print("[warn] 未指定 --cache-dir:走直读 h5 的慢路径。强烈建议先跑 prep_cache.py", flush=True)

        sdir = cfg_data.dataset.static_dir
        land = torch.from_numpy(np.load(os.path.join(sdir, "land_mask.npy")).astype(np.float32))
        soil = torch.from_numpy(np.load(os.path.join(sdir, "soil_type.npy")).astype(np.float32))
        topo = torch.from_numpy(np.load(os.path.join(sdir, "topography.npy")).astype(np.float32))
        topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
        surface_mask = torch.stack([land, soil, topo], 0).unsqueeze(0).repeat(
            cfg_data.dataloader.batch_size, 1, 1, 1).to(dev)

        teacher = Pangu(img_size=cfg_data.dataset.img_size, patch_size=cfg.patch_size,
                        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                        window_size=cfg.window_size).to(dev).eval()
        tck = torch.load("data/checkpoints/model_bak.pth", map_location=f"cuda:{dev}", weights_only=False)
        teacher.load_state_dict(tck["model_state_dict"] if "model_state_dict" in tck else tck)
        for p in teacher.parameters():
            p.requires_grad_(False)
        del tck
        tp = sum(p.numel() for p in teacher.parameters())

    # ── 学生 ──
    student = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                              num_heads=heads, global_mode=args.gm, mlp_ratio=args.mlp_ratio,
                              use_checkpoint=bool(args.gckpt)).to(dev)
    ema = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                          num_heads=heads, global_mode=args.gm,
                          mlp_ratio=args.mlp_ratio).to(dev).eval()
    ema.load_state_dict(student.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[model] teacher={tp/1e6:.2f}M  student={ps/1e6:.2f}M ({ps/tp*100:.1f}%)  "
          f"embed={args.embed} depths={depths} heads={heads} patch={patch} "
          f"gm={args.gm} mlp_ratio={args.mlp_ratio}", flush=True)

    # ── 优化器 ──
    if args.opt == "muon":
        # ★参数分组必须与历史最优一致:只有【骨干里的 2D 权重】走 Muon,
        #   patch embedding / recovery / logit_scale 这些要留给 AdamW ——
        #   Muon 的正交化更新对 patch 投影这类结构不合适。
        #   (只判 p.ndim==2 是不够的,会把 patchembed/recovery 错误地喂给 Muon。)
        mp, ap_ = [], []
        for n, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and not any(k in n for k in
                                       ("patchembed", "recovery", "embed", "conserv", "logit_scale")):
                mp.append(p)
            else:
                ap_.append(p)
        opt_m = Muon(mp, lr=args.muon_lr, momentum=args.muon_mom)
        opt_a = torch.optim.AdamW(ap_, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
        opts = [opt_m, opt_a]
        print(f"[opt] Muon({len(mp)} 个骨干 2D 权重, lr={args.muon_lr}, mom={args.muon_mom}) "
              f"+ AdamW({len(ap_)} 个其余, lr={args.lr})", flush=True)
    else:
        opts = [torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  weight_decay=0.05, betas=(0.9, 0.95))]
        print(f"[opt] AdamW(lr={args.lr})", flush=True)
    if args.spec > 0:
        print(f"[spec] 频域蒸馏开启: {args.spec_mode} 谱损失, 系数 {args.spec}"
              f"(高空 65 通道全权重,地表按 0.25 弱权)", flush=True)
    if args.aug:
        print("[aug] 经度周期性 roll 增广已开启", flush=True)

    # ── 断点续训 ──
    ckpt_path = args.save + ".ckpt"
    start_ep, best = 0, 1e9
    rf = args.resume if (args.resume and os.path.exists(args.resume)) else \
         (ckpt_path if os.path.exists(ckpt_path) else "")
    if rf:
        rk = torch.load(rf, map_location=f"cuda:{dev}", weights_only=False)
        student.load_state_dict(rk["model_state_dict"], strict=False)
        ema.load_state_dict(rk.get("ema_state_dict", rk["model_state_dict"]), strict=False)
        start_ep = int(rk.get("epoch", 0))
        best = float(rk.get("best", 1e9))
        for o, s in zip(opts, rk.get("opt_state", [])):
            try: o.load_state_dict(s)
            except Exception: pass
        print(f"[resume] 从 {rf} 的 ep{start_ep} 继续(best={best:.4f})", flush=True)

    tcache = SampleCache(0 if FAST else args.cache_gb)     # 快速路径不需要内存缓存

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    def build(data):
        invar, outvar = data[0], data[1]
        isf = invar[:, :4].to(dev, torch.float32); iua = invar[:, 4:].to(dev, torch.float32)
        x = torch.concat([isf, surface_mask, iua], dim=1)       # [B,72,721,1440]
        tsf = outvar[:, :4].to(dev, torch.float32); tua = outvar[:, 4:].to(dev, torch.float32)
        return x, tsf, tua

    def teach(x):
        ts, tu = teacher(x)
        return ts, tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3])

    vcache = SampleCache(0 if FAST else 4.0)
    vorder = []

    @torch.no_grad()
    def evaluate(model):
        model.eval(); gt = rel = 0.0; n = 0

        if FAST:                                           # 从预处理缓存读,不跑教师
            for item in val_loader:
                x, tsf, tua, ts, tu = dequant_batch(item, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ss, su = model(x)
                su = su.reshape(tua.shape)
                ss, su = ss.float(), su.float()
                gt += (wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)).item()
                num = (ss - ts).abs().mean() + (su - tu).abs().mean()
                den = ts.abs().mean() + tu.abs().mean() + 1e-6
                rel += (1 - (num / den)).item(); n += 1
                del x, tsf, tua, ts, tu
            return gt / max(1, n), rel / max(1, n)

        # 慢路径:验证样本进独立小缓存(固定住,否则每轮重跑教师且指标不可比)
        if vorder and all(k in vcache.store for k in vorder):
            items = [(k, None) for k in vorder]
        else:
            vorder.clear()
            got_src = []
            for vi, data in enumerate(val_loader):
                if args.max_val and vi >= args.max_val:
                    break
                got_src.append(data)
            items = [(sample_key(d), d) for d in got_src]

        for key, data in items:
            got = vcache.get(key, dev)
            if got is None:
                x, tsf, tua = build(data)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ts, tu = teach(x)
                ts, tu = ts.float(), tu.float()
                vcache.put(key, x, tsf, tua, ts, tu)
                if key is not None and key in vcache.store and key not in vorder:
                    vorder.append(key)
            else:
                x, tsf, tua, ts, tu = got
                if key is not None and key not in vorder:
                    vorder.append(key)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ss, su = model(x)
            su = su.reshape(tua.shape)
            ss, su = ss.float(), su.float()
            gt += (wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)).item()
            num = (ss - ts).abs().mean() + (su - tu).abs().mean()
            den = ts.abs().mean() + tu.abs().mean() + 1e-6
            rel += (1 - (num / den)).item(); n += 1
            del x, tsf, tua, ts, tu
        return gt / max(1, n), rel / max(1, n)

    def save_cfg():
        # ★ 必须把 global_mode / mlp_ratio 一并写入:inference.py 是按 cfg 重建模型的,
        #   缺任何一项都会退回默认值(mlp_ratio 默认 4.0),导致结构与权重对不上。
        return {"embed": args.embed, "depths": list(depths), "heads": list(heads),
                "patch": list(patch), "global_mode": args.gm, "mlp_ratio": args.mlp_ratio,
                "residual": False, "phys": 0, "n_phys": 0}

    import random as _rnd

    def step(x, tsf, tua, ts, tu):
        # 经度增广:地球经度是周期的,整体 roll 一个随机位移不改变物理,是免费的数据增强。
        # 输入、真值、教师输出必须 roll 同样的位移,否则对不齐。
        if args.aug:
            s = _rnd.randint(0, x.shape[-1] - 1)
            x = torch.roll(x, s, -1)
            tsf = torch.roll(tsf, s, -1); tua = torch.roll(tua, s, -1)
            ts = torch.roll(ts, s, -1);   tu = torch.roll(tu, s, -1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            ss, su = student(x)
        su = su.reshape(tua.shape)
        ss, su = ss.float(), su.float()
        ld = wl1(ss, ts, sw, 0.25) + wl1(su, tu, pw, 1.0)       # 对教师(蒸馏)
        lg = wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)     # 对 ERA5 真值(辅助)
        loss = args.alpha * ld + (1 - args.alpha) * lg

        # 频域蒸馏:对齐师生的功率谱,治学生输出过平滑(高空 65 通道为主,地表按 0.25 弱权)
        if args.spec > 0:
            _sl = spherical_spectral_loss if args.spec_mode == "sphere" else spectral_loss
            loss = loss + _sl(su, tu, args.spec) + _sl(ss, ts, args.spec * 0.25)

        for o in opts: o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for o in opts: o.step()
        ema_update(ema, student)
        return loss.item()

    # ── 训练循环 ────────────────────────────────────────────────────────────────
    # 快速路径(--cache-dir):每个 epoch 从预处理缓存里【随机遍历全部样本】。
    #   这是标准 SGD —— 梯度是整个训练分布的无偏估计,不存在灾难性遗忘。
    #   (曾考虑过"内存池轮换":灌一池训 N 轮再换下一批。它能绕开内存墙,但连续多轮只看同一批
    #    样本会让梯度有偏、模型过拟合当前池并遗忘前一池,而且余弦退火到后期 lr 很小时换上新池
    #    等于白灌。既然预处理落盘能做到读盘只要 0.05s,就没有任何理由冒这个险。)
    if FAST:
        n_train = len(train_loader.dataset)
        print(f"[train] 每个 epoch 随机遍历全部 {n_train} 个样本(标准 SGD,梯度无偏、无遗忘)",
              flush=True)

    for ep in range(start_ep, args.epochs):
        for o in opts:
            for g in o.param_groups:
                g["lr"] = (args.muon_lr if (args.opt == "muon" and o is opts[0]) else args.lr) * lr_at(ep)
        student.train(); t0 = time.time(); run = 0.0; c = 0

        if FAST:
            for item in train_loader:                       # worker 预取,读盘被计算掩盖
                x, tsf, tua, ts, tu = dequant_batch(item, dev)
                run += step(x, tsf, tua, ts, tu); c += 1
                del x, tsf, tua, ts, tu
        else:
            for j, data in enumerate(train_loader):         # 慢路径:直读 h5 + 实时跑教师
                if args.max_iters and j >= args.max_iters:
                    break
                key = sample_key(data)
                got = tcache.get(key, dev)
                if got is None:
                    x, tsf, tua = build(data)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        ts, tu = teach(x)
                    ts, tu = ts.float(), tu.float()
                    tcache.put(key, x, tsf, tua, ts, tu)
                else:
                    x, tsf, tua, ts, tu = got
                run += step(x, tsf, tua, ts, tu); c += 1
                del x, tsf, tua, ts, tu

        vgt, vrel = evaluate(ema)
        print(f"[ep{ep+1}/{args.epochs}] lr={args.lr*lr_at(ep):.2e} "
              f"train_loss={run/max(1,c):.4f} val_gtL1={vgt:.4f} val_match={vrel*100:.1f}% "
              f"| {c} 样本 | {(time.time()-t0):.0f}s", flush=True)

        torch.save({"model_state_dict": student.state_dict(), "ema_state_dict": ema.state_dict(),
                    "opt_state": [o.state_dict() for o in opts], "cfg": save_cfg(),
                    "epoch": ep + 1, "best": best, "val_match": vrel}, ckpt_path)
        if vgt < best:
            best = vgt
            torch.save({"model_state_dict": ema.state_dict(), "cfg": save_cfg(),
                        "val_gtL1": vgt, "val_match": vrel, "epoch": ep + 1}, args.save)
            print(f"  [saved best] val_gtL1={vgt:.4f} val_match={vrel*100:.1f}% -> {args.save}", flush=True)
    print(f"[done] best val_gtL1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
