# -*- coding: utf-8 -*-
"""
预处理:把 ERA5 样本 + 教师输出 一次性算好、int8 量化后落盘。
================================================================================
为什么要这一步(实测数据支撑):

  裸读一个 411MB 的 h5 只要 0.14 秒(NFS 带宽 2836 MB/s,一点不慢),
  但 dataloader 吐出一个样本却要 4.5 秒 —— 慢的不是磁盘,是 h5py:
  文件里有 99 个通道而模型只用 69 个,做通道选择时逐块 seek/解压,
  把 0.14 秒的顺序读拖成了 4.5 秒。

  拆开一个训练 iter:
      读盘(h5py)   4.4~6.1 s   ← 85%
      教师前向        0.80 s
      int8 量化       0.06 s
      学生前反向      0.24 s

  教师对同一样本的输出是常量,只需算一次;h5 的解析也只需做一次。
  于是把 (输入 x, ERA5 真值, 教师输出) 三份一起算好、int8 量化后存成 .npz:

      每样本 216 MB(fp16 需 435 MB) → 从盘里读回只要 0.05 秒(4.4 GB/s)

  训练时 4 个 worker 预取,读盘时间被 GPU 计算(0.24s)完全掩盖,
  且【每个 epoch 都能随机遍历全部样本】—— 标准 SGD,梯度无偏、不存在遗忘。
  内存里一个样本都不用常驻,彻底绕开容器 59GB 的 cgroup 上限。

存放位置:默认 /public/home(持久化,读 4.3 GB/s)。不要用 /tmp —— 那是容器的
overlay,容器一被回收就全没了(实测被回收过)。

用法(在 pangu_weather/ 下):
    python prep_cache.py --n 900 --out /public/home/xdzs2026_c296/cache_int8
    # 中断后重跑会自动跳过已存在的样本,可断点续做
================================================================================
"""
import os, sys, time, argparse
import numpy as np, torch
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams


def quant(t):
    """逐通道 |max|/127 定标的 int8 量化。都是归一化后的气象场,不需要 zero-point。
    相对误差约 0.4%,而蒸馏用 L1 损失、教师输出自身也带误差 —— 该噪声可忽略。"""
    a = t.detach().float()
    s = a.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / 127.0
    q = (a / s).round().clamp_(-127, 127).to(torch.int8)
    return q.cpu().numpy(), s.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_train.yaml")
    ap.add_argument("--n", type=int, default=900, help="预处理多少个训练样本")
    ap.add_argument("--nval", type=int, default=24, help="预处理多少个验证样本")
    ap.add_argument("--out", default="/public/home/xdzs2026_c296/cache_int8",
                    help="落盘目录。用 /public/home(持久化);别用 /tmp(容器回收即丢)")
    args = ap.parse_args()

    dev = 0
    cur = os.getcwd(); sys.path.append(cur)
    cfg = YParams(args.config, "model")
    cfg_data = YParams(args.config, "datapipe")
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)

    dp = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _ = dp.train_dataloader()
    val_loader, _ = dp.val_dataloader()

    sdir = cfg_data.dataset.static_dir
    land = torch.from_numpy(np.load(os.path.join(sdir, "land_mask.npy")).astype(np.float32))
    soil = torch.from_numpy(np.load(os.path.join(sdir, "soil_type.npy")).astype(np.float32))
    topo = torch.from_numpy(np.load(os.path.join(sdir, "topography.npy")).astype(np.float32))
    topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
    sm = torch.stack([land, soil, topo], 0).unsqueeze(0).to(dev)

    teacher = Pangu(img_size=cfg_data.dataset.img_size, patch_size=cfg.patch_size,
                    embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                    window_size=cfg.window_size).to(dev).eval()
    tck = torch.load("data/checkpoints/model_bak.pth", map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(tck["model_state_dict"] if "model_state_dict" in tck else tck)
    for p in teacher.parameters():
        p.requires_grad_(False)
    del tck
    print(f"[teacher] 官方 Pangu 全量权重已加载并冻结 ({sum(p.numel() for p in teacher.parameters())/1e6:.2f}M)",
          flush=True)

    @torch.no_grad()
    def do(loader, n, sub):
        outdir = os.path.join(args.out, sub)
        done = skip = 0
        t0 = time.time()
        for data in loader:
            if done + skip >= n:
                break
            try:
                key = str(data[4][-1][0])
            except Exception:
                key = "s%06d" % (done + skip)
            fp = os.path.join(outdir, key + ".npz")
            if os.path.exists(fp):                     # 断点续做
                skip += 1
                continue

            invar, outvar = data[0], data[1]
            isf = invar[:, :4].to(dev, torch.float32)
            iua = invar[:, 4:].to(dev, torch.float32)
            x = torch.concat([isf, sm, iua], dim=1)    # [1,72,721,1440]
            tsf = outvar[:, :4].to(dev, torch.float32)
            tua = outvar[:, 4:].to(dev, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ts, tu = teacher(x)
            tu = tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3])
            ts, tu = ts.float(), tu.float()

            qx, sx = quant(x); qa, sa = quant(tsf); qb, sb = quant(tua)
            qc, sc = quant(ts); qd, sd = quant(tu)
            np.savez(fp, x=qx, xs=sx, tsf=qa, tsfs=sa, tua=qb, tuas=sb,
                     ts=qc, tss=sc, tu=qd, tus=sd)
            done += 1
            del x, tsf, tua, ts, tu
            if done % 20 == 0:
                el = time.time() - t0
                gb = sum(os.path.getsize(os.path.join(outdir, f))
                         for f in os.listdir(outdir)) / (1 << 30)
                print("  [%s] %d/%d  %.1fs/样本  已占 %.1fGB  预计还需 %.0f 分钟"
                      % (sub, done + skip, n, el / max(1, done),
                         gb, (n - done - skip) * el / max(1, done) / 60), flush=True)
        return done, skip

    print(f"[prep] 训练样本 → {args.out}/train", flush=True)
    d, s = do(train_loader, args.n, "train")
    print(f"[prep] train 新增 {d} 个,跳过(已存在) {s} 个", flush=True)
    print(f"[prep] 验证样本 → {args.out}/val", flush=True)
    d, s = do(val_loader, args.nval, "val")
    print(f"[prep] val 新增 {d} 个,跳过 {s} 个", flush=True)

    for sub in ("train", "val"):
        p = os.path.join(args.out, sub)
        n = len(os.listdir(p))
        gb = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / (1 << 30)
        print(f"[done] {sub}: {n} 个样本, {gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
