# -*- coding: utf-8 -*-
"""
教师-学生蒸馏训练(SCNet 海光 DCU · 真实 ERA5)
- 数据/掩膜/损失完全沿用官方 train.py 口径(ERA5Datapipe + 3静态掩膜插第4位 + 加权L1 0.25/1.0)
- 模型换成 MaxVit3DStudent;教师 = 官方 Pangu(载 model_bak.pth, 冻结)
- 损失 = alpha·蒸馏(对教师输出) + (1-alpha)·对真值
在 pangu_weather/ 目录运行:
  python distill_train.py --embed 96 --depths 1,3,1 --heads 6,12,6 --epochs 1 --max-iters 5   # smoke
"""
import torch, os, sys, time, argparse, numpy as np
import torch.nn.functional as F
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from maxvit3d_student import MaxVit3DStudent


def loss_func(x, y, w, lw):
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=96)
    ap.add_argument("--depths", type=str, default="1,3,1")
    ap.add_argument("--heads", type=str, default="6,12,6")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=0, help=">0 限制每epoch迭代数(快速验证)")
    ap.add_argument("--alpha", type=float, default=0.7, help="蒸馏权重;(1-alpha)=对真值权重")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save", type=str, default="data/checkpoints/student.pth")
    args = ap.parse_args()
    depths = tuple(int(v) for v in args.depths.split(","))
    heads = tuple(int(v) for v in args.heads.split(","))

    dev = 0
    torch.backends.cuda.matmul.allow_tf32 = True
    cur = os.getcwd(); sys.path.append(cur)
    cfgp = os.path.join(cur, "conf/config.yaml")
    cfg = YParams(cfgp, "model")
    cfg_data = YParams(cfgp, "datapipe")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _ = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()
    print(f"[data] train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)

    sw = torch.as_tensor(cfg_data.dataset.weights[:4], device=dev, dtype=torch.float32).view(1, -1, 1, 1)
    pw = torch.as_tensor(cfg_data.dataset.weights[4:], device=dev, dtype=torch.float32).view(1, -1, 1, 1)

    sdir = cfg_data.dataset.static_dir
    land = torch.from_numpy(np.load(os.path.join(sdir, "land_mask.npy")).astype(np.float32))
    soil = torch.from_numpy(np.load(os.path.join(sdir, "soil_type.npy")).astype(np.float32))
    topo = torch.from_numpy(np.load(os.path.join(sdir, "topography.npy")).astype(np.float32))
    topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land, soil, topo], dim=0).unsqueeze(0).repeat(
        cfg_data.dataloader.batch_size, 1, 1, 1).to(dev)

    teacher = Pangu(img_size=cfg_data.dataset.img_size, patch_size=cfg.patch_size,
                    embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                    window_size=cfg.window_size).to(dev).eval()
    ck = torch.load("data/checkpoints/model_bak.pth", map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(ck["model_state_dict"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] Pangu loaded & frozen", flush=True)

    student = MaxVit3DStudent(img_size=tuple(cfg_data.dataset.img_size), embed_dim=args.embed,
                              depths=depths, num_heads=heads, use_checkpoint=True).to(dev)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[student] embed={args.embed} depths={depths} heads={heads} params={ps/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)

    def build(invar, outvar):
        isf = invar[:, :4].to(dev, torch.float32)
        iua = invar[:, 4:].to(dev, torch.float32)
        x = torch.concat([isf, surface_mask, iua], dim=1)
        tsf = outvar[:, :4].to(dev, torch.float32)
        tua = outvar[:, 4:].to(dev, torch.float32)
        return x, tsf, tua

    for ep in range(args.epochs):
        student.train()
        t0 = time.time(); run = 0.0; cnt = 0
        for j, data in enumerate(train_loader):
            if args.max_iters and j >= args.max_iters:
                break
            x, tsf, tua = build(data[0], data[1])
            with torch.no_grad():
                ts, tu = teacher(x); tu = tu.reshape(tua.shape)
            ss, su = student(x); su = su.reshape(tua.shape)
            ld = loss_func(ss, ts, sw, 0.25) + loss_func(su, tu, pw, 1.0)
            lg = loss_func(ss, tsf, sw, 0.25) + loss_func(su, tua, pw, 1.0)
            loss = args.alpha * ld + (1 - args.alpha) * lg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item(); cnt += 1
            print(f"[ep{ep} it{j+1}] loss={loss.item():.4f} distill={ld.item():.4f} gt={lg.item():.4f} "
                  f"{(time.time()-t0)/(j+1):.1f}s/it", flush=True)

        student.eval(); vl = 0.0; n = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                if args.max_iters and j >= max(2, args.max_iters // 4):
                    break
                x, tsf, tua = build(data[0], data[1])
                ss, su = student(x); su = su.reshape(tua.shape)
                vl += (loss_func(ss, tsf, sw, 0.25) + loss_func(su, tua, pw, 1.0)).item(); n += 1
        print(f"[ep{ep}] train_loss={run/max(1,cnt):.4f} val_gtL1={vl/max(1,n):.4f}", flush=True)
        os.makedirs("data/checkpoints", exist_ok=True)
        torch.save({"model_state_dict": student.state_dict(),
                    "cfg": {"embed": args.embed, "depths": list(depths), "heads": list(heads)}}, args.save)
        print(f"[saved] {args.save}", flush=True)


if __name__ == "__main__":
    main()
