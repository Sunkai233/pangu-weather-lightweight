# -*- coding: utf-8 -*-
"""落盘缓存蒸馏(SCNet 海光 DCU · 适配 63.8G 容器内存上限)——真实共享仓库数据 · 几百样本。
内存优化:教师输出【一次性预计算落盘】(每样本存 输入69ch + 教师69ch, fp16),训练用 DataLoader
         按需从磁盘读(RAM 只占少量),而非 distill_scnet_fast 那样全塞进 RAM(几百样本会 OOM)。
数据:era5_big(从 /work2 共享盘拷来的 1980/1981,99ch→按 metadata 过滤 69ch,已归一)。
损失:纯蒸馏对教师(alpha=1,省去真值缓存);验证集留 RAM(带真值)算 val_gtL1 + val_match。
用法(_bigtrain/ 下):python distill_diskcache_scnet.py --limit-train 250 --epochs 8 --embed 64 --patch 2,8,8
"""
import os, sys, time, math, argparse
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
import maxvit3d_student as M
from maxvit3d_student import MaxVit3DStudent

DEV = 0


def wl1(x, y, w, lw):
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


@torch.no_grad()
def ema_update(ema, model, d=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


class DiskCache(Dataset):
    """按需从磁盘读一个样本(输入69 + 教师sf/ua),低内存。"""
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location="cpu")
        return d["x69"], d["tsf"], d["tua"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--depths", type=str, default="1,3,1")
    ap.add_argument("--heads", type=str, default="4,8,4")
    ap.add_argument("--patch", type=str, default="2,8,8")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=250)
    ap.add_argument("--cache-dir", type=str, default="/public/home/xdzs2026_c296/_bigtrain/tcache")
    ap.add_argument("--save", type=str, default="/public/home/xdzs2026_c296/_bigtrain/student_big.pth")
    args = ap.parse_args()
    depths = tuple(int(v) for v in args.depths.split(","))
    heads = tuple(int(v) for v in args.heads.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))
    M.set_sdpa(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    cur = os.getcwd(); sys.path.append(cur)
    cfgp = os.path.join(cur, "conf/config.yaml")
    cfg = YParams(cfgp, "model"); cfg_data = YParams(cfgp, "datapipe")
    print(f"[cfg] embed={args.embed} patch={patch} epochs={args.epochs} limit_train={args.limit_train}", flush=True)

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _ = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()

    sw = torch.as_tensor(cfg_data.dataset.weights[:4], device=DEV, dtype=torch.float32).view(1, -1, 1, 1)
    pw = torch.as_tensor(cfg_data.dataset.weights[4:], device=DEV, dtype=torch.float32).view(1, -1, 1, 1)
    sdir = cfg_data.dataset.static_dir
    land = torch.from_numpy(np.load(os.path.join(sdir, "land_mask.npy")).astype(np.float32))
    soil = torch.from_numpy(np.load(os.path.join(sdir, "soil_type.npy")).astype(np.float32))
    topo = torch.from_numpy(np.load(os.path.join(sdir, "topography.npy")).astype(np.float32))
    topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
    smask = torch.stack([land, soil, topo], 0).unsqueeze(0).to(DEV)  # 1,3,H,W 静态掩码(常量)

    teacher = Pangu(img_size=cfg_data.dataset.img_size, patch_size=cfg.patch_size,
                    embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                    window_size=cfg.window_size).to(DEV).eval()
    ck = torch.load("data/checkpoints/model_bak.pth", map_location=f"cuda:{DEV}", weights_only=False)
    teacher.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded & frozen", flush=True)

    def build72(invar):
        isf = invar[:, :4].to(DEV, torch.float32); iua = invar[:, 4:].to(DEV, torch.float32)
        x = torch.cat([isf, smask, iua], 1)          # 72ch 学生/教师输入
        x69 = torch.cat([isf, iua], 1)               # 69ch(去掉静态掩码,存盘用,训练时再插回)
        return x, x69

    # ---------- 落盘缓存:训练集(输入69 + 教师69),纯蒸馏,不存真值 ----------
    os.makedirs(args.cache_dir, exist_ok=True)
    paths = []; t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if i >= args.limit_train:
                break
            x, x69 = build72(data[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ts, tu = teacher(x)
            tu = tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3])
            fp = os.path.join(args.cache_dir, f"tr_{i:04d}.pt")
            torch.save({"x69": x69[0].half().cpu(), "tsf": ts[0].half().cpu(), "tua": tu[0].half().cpu()}, fp)
            paths.append(fp)
            if (i + 1) % 50 == 0:
                print(f"  [cache] {i+1}/{args.limit_train}  {(time.time()-t0):.0f}s", flush=True)
    print(f"[cache] train={len(paths)} 落盘完成 用时{(time.time()-t0):.0f}s  dir={args.cache_dir}", flush=True)

    # 验证集留 RAM(带真值,39个,~17GB)
    val_cache = []
    with torch.no_grad():
        for data in val_loader:
            x, _ = build72(data[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ts, tu = teacher(x)
            tu = tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3])
            tsf_t = data[1][:, :4].to(DEV, torch.float32); tua_t = data[1][:, 4:].to(DEV, torch.float32)
            val_cache.append((x.half().cpu(), ts.half().cpu(), tu.half().cpu(), tsf_t.half().cpu(), tua_t.half().cpu()))
    print(f"[cache] val={len(val_cache)} 留RAM", flush=True)
    del teacher; torch.cuda.empty_cache()

    student = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                              num_heads=heads, use_checkpoint=False).to(DEV)
    ema = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                          num_heads=heads).to(DEV).eval()
    ema.load_state_dict(student.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[model] student={ps/1e6:.2f}M ({ps/64.22e6*100:.1f}%)", flush=True)

    loader = DataLoader(DiskCache(paths), batch_size=1, shuffle=True, num_workers=3,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    def ins_mask(x69):  # 69ch -> 72ch 插回静态掩码
        return torch.cat([x69[:, :4], smask, x69[:, 4:]], 1)

    @torch.no_grad()
    def evaluate(model):
        model.eval(); gt = 0.0; rel = 0.0; n = 0
        for x, ts, tu, tsf, tua in val_cache:
            x = x.float().to(DEV); ts = ts.float().to(DEV); tu = tu.float().to(DEV)
            tsf = tsf.float().to(DEV); tua = tua.float().to(DEV)
            ss, su = model(x); su = su.reshape(tua.shape)
            gt += (wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)).item()
            num = (ss - ts).abs().mean() + (su - tu).abs().mean()
            den = ts.abs().mean() + tu.abs().mean() + 1e-6
            rel += (1 - (num / den)).item(); n += 1
        return gt / max(1, n), rel / max(1, n)

    best = 1e9
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(ep)
        student.train(); t0 = time.time(); run = 0.0; c = 0
        for x69, tsf, tua in loader:
            x = ins_mask(x69.float().to(DEV))
            ts = tsf.float().to(DEV); tu = tua.float().to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ss, su = student(x); su = su.reshape(tu.shape)
            loss = wl1(ss.float(), ts, sw, 0.25) + wl1(su.float(), tu, pw, 1.0)  # 纯蒸馏
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); ema_update(ema, student)
            run += loss.item(); c += 1
        vgt, vrel = evaluate(ema)
        print(f"[ep{ep+1}/{args.epochs}] lr={args.lr*lr_at(ep):.2e} train={run/max(1,c):.4f} "
              f"val_gtL1={vgt:.4f} val_match={vrel*100:.1f}% {(time.time()-t0):.0f}s", flush=True)
        if vgt < best:
            best = vgt
            torch.save({"model_state_dict": ema.state_dict(),
                        "cfg": {"embed": args.embed, "depths": list(depths), "heads": list(heads), "patch": list(patch)},
                        "val_gtL1": vgt, "val_match": vrel, "epoch": ep + 1}, args.save)
            print(f"  [saved best] val_gtL1={vgt:.4f} val_match={vrel*100:.1f}%", flush=True)
    print(f"[done] best val_gtL1={best:.4f}  student={args.save}", flush=True)


if __name__ == "__main__":
    main()
