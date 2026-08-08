# -*- coding: utf-8 -*-
"""
快速完整教师-学生微调(SCNet 海光 DCU)——为 DCU 调优:
关键加速:① 教师输出 + 输入 + 真值 开局一次性缓存进内存(节点461G RAM),训练循环不再跑教师(省795ms/step)、不再读NFS;
         ② 关梯度检查点(64GB显存够);③ 材料化注意力(DCU上SDPA=math反而略慢)。
→ step 从 ~1900ms 降到 ~900ms,79样本 ~70s/epoch,60ep 约70分钟可在2h实例内收敛。
损失:alpha·蒸馏(教师) + (1-alpha)·真值(真实6h)。框架:AdamW+warmup余弦+裁剪+EMA+bf16+最优检查点。
用法(pangu_weather/ 下):python distill_scnet_fast.py --embed 64 --patch 2,8,8 --epochs 60 --alpha 0.5
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
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


@torch.no_grad()
def ema_update(ema, model, d=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--depths", type=str, default="1,3,1")
    ap.add_argument("--heads", type=str, default="4,8,4")
    ap.add_argument("--patch", type=str, default="2,8,8")
    ap.add_argument("--sdpa", type=int, default=0)  # DCU 上 materialized 略快,默认关
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--save", type=str, default="/public/home/xdzs2026_c296/student_scnet_fast.pth")
    args = ap.parse_args()
    depths = tuple(int(v) for v in args.depths.split(","))
    heads = tuple(int(v) for v in args.heads.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))
    dev = 0
    torch.backends.cuda.matmul.allow_tf32 = True
    M.set_sdpa(bool(args.sdpa))
    cur = os.getcwd(); sys.path.append(cur)
    cfgp = os.path.join(cur, "conf/config.yaml")
    cfg = YParams(cfgp, "model"); cfg_data = YParams(cfgp, "datapipe")
    print(f"[cfg] embed={args.embed} patch={patch} sdpa={args.sdpa} alpha={args.alpha} epochs={args.epochs}", flush=True)

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _ = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()

    sw = torch.as_tensor(cfg_data.dataset.weights[:4], device=dev, dtype=torch.float32).view(1, -1, 1, 1)
    pw = torch.as_tensor(cfg_data.dataset.weights[4:], device=dev, dtype=torch.float32).view(1, -1, 1, 1)
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
    ck = torch.load("data/checkpoints/model_bak.pth", map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded & frozen", flush=True)

    def build(data):
        invar, outvar = data[0], data[1]
        isf = invar[:, :4].to(dev, torch.float32); iua = invar[:, 4:].to(dev, torch.float32)
        x = torch.concat([isf, surface_mask, iua], dim=1)
        tsf = outvar[:, :4].to(dev, torch.float32); tua = outvar[:, 4:].to(dev, torch.float32)
        return x, tsf, tua

    # ---------- 一次性缓存:input + 教师输出 + 真值(CPU fp16) ----------
    def cache_set(loader):
        c = []; t0 = time.time()
        with torch.no_grad():
            for data in loader:
                x, tsf, tua = build(data)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ts, tu = teacher(x)
                tu = tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3])
                c.append((x.half().cpu(), ts.half().cpu(), tu.half().cpu(), tsf.half().cpu(), tua.half().cpu()))
        return c, time.time() - t0
    train_cache, tt = cache_set(train_loader)
    val_cache, _ = cache_set(val_loader)
    print(f"[cache] train={len(train_cache)} val={len(val_cache)}  教师预算用时{tt:.0f}s(只此一次)", flush=True)
    del teacher; torch.cuda.empty_cache()

    student = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                              num_heads=heads, use_checkpoint=False).to(dev)
    ema = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths,
                          num_heads=heads).to(dev).eval()
    ema.load_state_dict(student.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[model] student={ps/1e6:.2f}M ({ps/64.22e6*100:.1f}%)", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def evaluate(model):
        model.eval(); gt = 0.0; rel = 0.0; n = 0
        for x, ts, tu, tsf, tua in val_cache:
            x = x.float().to(dev); ts = ts.float().to(dev); tu = tu.float().to(dev)
            tsf = tsf.float().to(dev); tua = tua.float().to(dev)
            ss, su = model(x); su = su.reshape(tua.shape)
            gt += (wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)).item()
            num = (ss - ts).abs().mean() + (su - tu).abs().mean()
            den = ts.abs().mean() + tu.abs().mean() + 1e-6
            rel += (1 - (num / den)).item(); n += 1
        return gt / max(1, n), rel / max(1, n)

    import random
    best = 1e9
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(ep)
        student.train(); t0 = time.time(); run = 0.0; c = 0
        random.shuffle(train_cache)
        for x, ts, tu, tsf, tua in train_cache:
            x = x.float().to(dev); ts = ts.float().to(dev); tu = tu.float().to(dev)
            tsf = tsf.float().to(dev); tua = tua.float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ss, su = student(x); su = su.reshape(tua.shape)
            ld = wl1(ss.float(), ts, sw, 0.25) + wl1(su.float(), tu, pw, 1.0)
            lg = wl1(ss.float(), tsf, sw, 0.25) + wl1(su.float(), tua, pw, 1.0)
            loss = args.alpha * ld + (1 - args.alpha) * lg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            ema_update(ema, student)
            run += loss.item(); c += 1
        vgt, vrel = evaluate(ema)
        print(f"[ep{ep+1}/{args.epochs}] lr={args.lr*lr_at(ep):.2e} train={run/max(1,c):.4f} "
              f"val_gtL1={vgt:.4f} val_match={vrel*100:.1f}% {(time.time()-t0):.0f}s", flush=True)
        if vgt < best:
            best = vgt
            torch.save({"model_state_dict": ema.state_dict(),
                        "cfg": {"embed": args.embed, "depths": list(depths), "heads": list(heads),
                                "patch": list(patch), "sdpa": args.sdpa},
                        "val_gtL1": vgt, "val_match": vrel, "epoch": ep + 1}, args.save)
            print(f"  [saved best] val_gtL1={vgt:.4f} val_match={vrel*100:.1f}%", flush=True)
    print(f"[done] best val_gtL1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
