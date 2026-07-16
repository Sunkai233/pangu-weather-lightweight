# -*- coding: utf-8 -*-
"""DeepLN 高效蒸馏:教师输出一次性缓存进 RAM(376G),之后学生训练不跑教师(快~10x)+ 经度增广 + fp16(V100)。
数据:/data/era5_full(99通道密集 ERA5,从 SCNet /work2 拉取)→ 切片成 69 → 拼掩膜成 72 → 教师生成蒸馏目标。
用法(/root 下):python distill_cache.py --train-years 1980,...,1998 --val-years 1999,2000 --embed 64 --patch 2,8,8 --epochs 40 --amp fp16
"""
import os, glob, time, math, argparse, random
import numpy as np, h5py, torch
import torch.nn.functional as F
from onescience.models.pangu import Pangu
from maxvit3d_student import MaxVit3DStudent
import maxvit3d_student as _M

CKPT = os.environ.get("PCKPT", "/data2/sunkai/assets/model_bak_conv.pth")
STATS = os.environ.get("PSTATS", "/data2/sunkai/assets/ERA5_test")
META99 = os.environ.get("PMETA", "/data2/sunkai/assets/era5_real_meta.json")
BASE = os.environ.get("PBASE", "/data2/sunkai/era5_full")
import os as _os
if _os.environ.get("SAVETEMP", "0") == "1":  # 救温度:2m_t+近地面temperature/geopotential权重↑(拖W通道)
    WEIGHTS = [1.50, 0.77, 0.66, 5.00] + [4.0,3.5,3.0]+[3.0]*10 + [0.6]*13 + [4.0,3.5,3.0,2.5,2.5,2.5,2.0,1.5,1.5,1.0,1.0,0.8,0.6] + [0.77]*13 + [0.54]*13
else:
    WEIGHTS = [1.50, 0.77, 0.66, 3.00] + [3.0]*13 + [0.6]*13 + [1.5]*13 + [0.77]*13 + [0.54]*13
_LEV = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CONFIG69 = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"]
for _v in ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]:
    CONFIG69 += [f"{_v}_{_l}" for _l in _LEV]


def wl1(x, y, w, lw):
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


def spectral_loss(pred, target, lw):
    """频域蒸馏(平面FFT版):对齐师生输出的 2D 功率谱(log-power L1)。
    学生过平滑→高频功率偏低→log-power 差被惩罚→逼学生补回高频细节,正面攻 L2 真值的过平滑失分。
    在 autocast 外用 float 计算(FFT 对 fp16 不稳)。"""
    pf = torch.fft.rfft2(pred.float(), norm="ortho")
    tf = torch.fft.rfft2(target.float(), norm="ortho")
    lp = torch.log(pf.real ** 2 + pf.imag ** 2 + 1e-6)
    lt = torch.log(tf.real ** 2 + tf.imag ** 2 + 1e-6)
    return lw * (lp - lt).abs().mean()


_SHT = None
def spherical_spectral_loss(pred, target, lw):
    """球面谱蒸馏(球谐SHT版,针对球面ERA5场):对齐师生的角功率谱 C_ℓ=Σ_m|a_ℓm|²。
    ERA5是球面场,平面FFT在极区(经线收敛)失真;球谐变换几何正确,C_ℓ 按球谐阶ℓ天然分尺度。
    学生过平滑→高ℓ(小尺度)功率偏低→log C_ℓ 差被罚→逼补高频,且极区不再被平面FFT误判。"""
    global _SHT
    if _SHT is None:
        from torch_harmonics import RealSHT
        H, W = pred.shape[-2], pred.shape[-1]
        _SHT = RealSHT(H, W, grid="equiangular").to(pred.device)
    pf = _SHT(pred.float()); tf = _SHT(target.float())          # (B,C,lmax,mmax) complex
    pc = (pf.real ** 2 + pf.imag ** 2).sum(dim=-1)              # 角功率 C_ℓ (B,C,lmax)
    tc = (tf.real ** 2 + tf.imag ** 2).sum(dim=-1)
    return lw * (torch.log(pc + 1e-8) - torch.log(tc + 1e-8)).abs().mean()


@torch.no_grad()
def ema_update(ema, model, d=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def build_files(years):
    files = []
    for y in years:
        files += sorted(glob.glob(f"{BASE}/data/{y}/*.h5"))
    return files


def read_x(f, mu, sd, masks, sel69):
    with h5py.File(f, "r") as h:
        fld = h["fields"][:].astype(np.float32)
    if fld.shape[0] != 69:
        fld = fld[sel69]
    fld = (fld - mu) / sd
    return np.concatenate([fld[:4], masks, fld[4:]], 0)  # (72,721,1440) fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--depths", type=str, default="1,3,1")
    ap.add_argument("--heads", type=str, default="4,8,4")
    ap.add_argument("--patch", type=str, default="2,8,8")
    ap.add_argument("--train-years", dest="ty", type=str, default="1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996")
    ap.add_argument("--val-years", dest="vy", type=str, default="1997,1998")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--residual", type=int, default=0, help="残差预测out=x_in+NN(persistence物理基线,网络只学增量,减负)")
    ap.add_argument("--ivw", type=int, default=0, help="增量逆方差:按每通道6h增量std倒数加权(GraphCast式,温度小增量自动放大梯度,突出难学小信号)")
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--sdpa", type=int, default=1)
    ap.add_argument("--amp", type=str, default="fp16")
    ap.add_argument("--max-cache", dest="maxc", type=int, default=1100)
    ap.add_argument("--global-mode", dest="gm", type=str, default="grid")
    ap.add_argument("--resume", type=str, default="", help="补充训练:从该checkpoint权重继续(不从头)")
    ap.add_argument("--save", type=str, default="/data2/sunkai/student_mitai.pth")
    ap.add_argument("--opt", type=str, default="adamw", choices=["adamw", "muon"], help="优化器:adamw 或 muon(2D权重走Muon其余AdamW)")
    ap.add_argument("--muon-lr", dest="muon_lr", type=float, default=0.02, help="Muon学习率(比AdamW大一量级)")
    ap.add_argument("--muon-momentum", dest="muon_momentum", type=float, default=0.95, help="Muon动量(0.95默认,0.99更平滑,优化器调优项)")
    ap.add_argument("--spec", type=float, default=0.0, help="频域蒸馏:谱对齐损失系数(治过平滑),0=关闭")
    ap.add_argument("--spec-mode", dest="spec_mode", type=str, default="plane", choices=["plane", "sphere"], help="谱损失模式:plane=平面FFT / sphere=球谐SHT(球面ERA5几何正确)")
    ap.add_argument("--mlp-ratio", dest="mlp_ratio", type=float, default=4.0, help="SwiGLU FFN宽度比(激活实测只用7-10%维度→可大降,如1.5/2.0,三吃轻量+显存+推理)")
    ap.add_argument("--freeze-backbone", dest="freeze_bb", type=int, default=0, help="冻结backbone只训stem/recovery:迁移mitai backbone到DCU SCNet API(backbone通用,只重训首尾patch投影适配新stem)")
    ap.add_argument("--dtp", type=int, default=0, help="DTP动态加权:难通道(对教师L1大,如近地面)自动加权救误差不均提W(逐epoch按per-channel L1 EMA调,clamp0.3-3均值1)")
    args = ap.parse_args()
    # 正式提交只开放最终权重实际采用的直接 y、纯教师蒸馏路径。
    if args.residual or args.ivw or args.spec or args.freeze_bb or args.dtp or _os.environ.get("SAVETEMP", "0") != "0":
        ap.error("正式复现配置固定为 residual=0, ivw=0, spec=0, freeze-backbone=0, dtp=0, SAVETEMP=0")
    depths = tuple(int(v) for v in args.depths.split(","))
    heads = tuple(int(v) for v in args.heads.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))
    dev = 0
    _M.set_sdpa(bool(args.sdpa))
    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]
    use_amp = args.amp != "fp32"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))
    torch.backends.cuda.matmul.allow_tf32 = True

    import json
    mu = np.load(f"{STATS}/stats/global_means.npy").astype(np.float32)[0]
    sd = np.load(f"{STATS}/stats/global_stds.npy").astype(np.float32)[0]
    land = np.load(f"{STATS}/static/land_mask.npy").astype(np.float32)
    soil = np.load(f"{STATS}/static/soil_type.npy").astype(np.float32)
    topo = np.load(f"{STATS}/static/topography.npy").astype(np.float32); topo = (topo - topo.mean()) / (topo.std() + 1e-6)
    masks = np.stack([land, soil, topo], 0)
    m99 = json.load(open(META99))["variables"]
    sel69 = np.array([m99.index(v) for v in CONFIG69])
    if mu.shape[0] != 69:  # stats是99ch全量(如DCU era5_real)→按sel69选成69,对齐read_x里选过的fld
        mu = mu[sel69]; sd = sd[sel69]
    sw = torch.tensor(WEIGHTS[:4], device=dev).view(1, 4, 1, 1)
    pw = torch.tensor(WEIGHTS[4:], device=dev).view(1, 65, 1, 1)
    _lat = torch.cos(torch.deg2rad(torch.linspace(90, -90, 721, device=dev)))
    coslat = (_lat / _lat.mean()).view(1, 1, 721, 1)  # 纬度面积加权(GraphCast标准,零成本)

    ty = [y.strip() for y in args.ty.split(",") if y.strip()]
    vy = [y.strip() for y in args.vy.split(",") if y.strip()]
    tr_files = build_files(ty); random.shuffle(tr_files); tr_files = tr_files[:args.maxc]
    va_files = build_files(vy)[:200]
    print(f"[data] train_files={len(tr_files)} val_files={len(va_files)} (tr={ty} va={vy})", flush=True)

    teacher = Pangu(img_size=(721, 1440)).to(dev).eval()
    ck = torch.load(CKPT, map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.half()
    print("[teacher] loaded", flush=True)

    def teach(x):
        with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
            ts, tu = teacher(x)
        return ts.float(), tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3]).float()

    def cache_files(files):  # 流式:逐文件读->教师->只存 fp16(避免 fp32 全塞内存 OOM)
        c = []; t0 = time.time(); skipped = 0
        with torch.no_grad():
            for i, f in enumerate(files):
                try:  # 容错:跳过 pull 并发写入的截断/坏 h5(训练与 pull 共享数据目录时必需)
                    x = torch.from_numpy(read_x(f, mu, sd, masks, sel69)).unsqueeze(0).to(dev)
                except Exception:
                    skipped += 1; continue
                ts, tu = teach(x)
                c.append((x.half().cpu(), ts.half().cpu(), tu.half().cpu()))
                if (i + 1) % 100 == 0:
                    print(f"  [cache] {i+1}/{len(files)} {(time.time()-t0):.0f}s", flush=True)
        if skipped:
            print(f"  [cache] 跳过 {skipped} 个坏/截断文件(pull并发写)", flush=True)
        return c, time.time() - t0
    tr, tt = cache_files(tr_files)
    va, _ = cache_files(va_files)
    print(f"[cache] train={len(tr)} val={len(va)} 教师预算{tt:.0f}s RAM≈{(len(tr)+len(va))*0.29:.0f}G", flush=True)
    if args.ivw:  # 增量逆方差:统计每通道6h增量(teacher-x_in)的std,1/std加权→温度小增量自动放大(残差比0.07-0.12)
        acc_s = torch.zeros(4); acc_u = torch.zeros(65); ns = 0
        for x, ts, tu in tr:
            inc_s = ts.float() - x[:, :4].float(); inc_u = tu.float() - x[:, 7:72].float()
            acc_s += (inc_s ** 2).mean(dim=(0, 2, 3)); acc_u += (inc_u ** 2).mean(dim=(0, 2, 3)); ns += 1
        ds_s = (acc_s / ns).sqrt(); ds_u = (acc_u / ns).sqrt()
        ivw_s = (1.0 / (ds_s + 1e-6)); ivw_s = (ivw_s / ivw_s.mean()).clamp(0.3, 3.0).to(dev)
        ivw_u = (1.0 / (ds_u + 1e-6)); ivw_u = (ivw_u / ivw_u.mean()).clamp(0.3, 3.0).to(dev)
        sw = sw * ivw_s.view(1, 4, 1, 1); pw = pw * ivw_u.view(1, 65, 1, 1)
        print(f"[ivw] 增量逆方差: surf权重{[round(v,2) for v in ivw_s.tolist()]} upper最重通道#{ivw_u.argmax().item()}({ivw_u.max():.2f})", flush=True)
    del teacher; torch.cuda.empty_cache()

    student = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths, num_heads=heads,
                              use_checkpoint=False, global_mode=args.gm, mlp_ratio=args.mlp_ratio, residual=args.residual).to(dev)
    ema = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths, num_heads=heads,
                          global_mode=args.gm, mlp_ratio=args.mlp_ratio, residual=args.residual).to(dev).eval()
    start_ep = 0; best = -1.0
    ckpt_path = args.save + ".ckpt"   # 每轮断点(防中断,重开自动续训)
    resume_from = args.resume if (args.resume and os.path.exists(args.resume)) else (ckpt_path if os.path.exists(ckpt_path) else "")
    if resume_from:  # 优先指定 --resume,否则自动检测断点 .ckpt 续训
        rk = torch.load(resume_from, map_location=f"cuda:{dev}", weights_only=False)
        rsd = rk["model_state_dict"] if "model_state_dict" in rk else rk
        student.load_state_dict(rsd, strict=True)
        start_ep = rk.get("epoch", 0); best = rk.get("best", rk.get("val_match", -1.0))
        print(f"[resume] 从 {resume_from} ep{start_ep} best={best*100:.2f}% 续训(strict=True)", flush=True)
    if args.freeze_bb:  # 迁移mitai backbone到DCU SCNet API:冻结backbone只训stem/recovery,从头训stem(不继承mitai epoch)
        nf = 0
        for n, p in student.named_parameters():
            if not any(s in n.lower() for s in ("patchembed", "recovery", "reconvery")):
                p.requires_grad_(False); nf += 1
        start_ep = 0; best = -1.0
        ntr = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"[freeze] 冻结backbone {nf}参数,只训stem/recovery({ntr/1e6:.2f}M可训),从头训stem适配迁移的backbone", flush=True)
    ema.load_state_dict(student.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[model] student={ps/1e6:.2f}M ({ps/64.22e6*100:.1f}%)", flush=True)
    if args.opt == "muon":
        from muon import Muon
        muon_p, adamw_p = [], []
        for n, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and not any(k in n for k in ("patchembed", "recovery", "embed", "conserv", "logit_scale")):
                muon_p.append(p)
            else:
                adamw_p.append(p)
        opt_m = Muon(muon_p, lr=args.muon_lr, momentum=args.muon_momentum)
        opt_a = torch.optim.AdamW(adamw_p, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
        opts = [(opt_m, args.muon_lr), (opt_a, args.lr)]
        print(f"[opt] Muon {len(muon_p)}个2D权重(lr={args.muon_lr}) + AdamW {len(adamw_p)}个其余(lr={args.lr})", flush=True)
    else:
        opt_a = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
        opts = [(opt_a, args.lr)]
        print(f"[opt] AdamW(lr={args.lr})", flush=True)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def evaluate(model):
        model.eval(); rel = 0.0; n = 0
        for x, ts, tu in va:
            x = x.float().to(dev); ts = ts.float().to(dev); tu = tu.float().to(dev)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                ss, su = model(x); su = su.reshape(tu.shape)
            ss, su = ss.float(), su.float()
            num = (ss - ts).abs().mean() + (su - tu).abs().mean()
            den = ts.abs().mean() + tu.abs().mean() + 1e-6
            rel += (1 - (num / den)).item(); n += 1
        return rel / max(1, n)

    dyn_w = torch.ones(69, device=dev)  # DTP动态权重(surface4+upper65):难通道(对教师L1大,近地面)自动加权救误差不均
    ch_l1_ema = None
    for ep in range(start_ep, args.epochs):   # best/start_ep 已在 resume 段设
        for _opt, _base in opts:
            for g in _opt.param_groups:
                g["lr"] = _base * lr_at(ep)
        student.train(); t0 = time.time(); run = 0.0; c = 0
        ch_l1_acc = torch.zeros(69, device=dev); ch_l1_n = 0
        random.shuffle(tr)
        for x, ts, tu in tr:
            x = x.float().to(dev); ts = ts.float().to(dev); tu = tu.float().to(dev)
            if args.aug:
                s = random.randint(0, x.shape[-1] - 1)
                x = torch.roll(x, s, -1); ts = torch.roll(ts, s, -1); tu = torch.roll(tu, s, -1)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                ss, su = student(x); su = su.reshape(tu.shape)
            loss = wl1(ss.float(), ts, sw * dyn_w[:4].view(1, 4, 1, 1) * coslat, 0.25) + wl1(su.float(), tu, pw * dyn_w[4:].view(1, 65, 1, 1) * coslat, 1.0)
            if args.dtp:
                with torch.no_grad():
                    ch_l1_acc += torch.cat([(ss.float() - ts).abs().mean(dim=(0, 2, 3)), (su.float() - tu).abs().mean(dim=(0, 2, 3))]); ch_l1_n += 1
            if args.spec > 0:  # 频域蒸馏:谱对齐治过平滑(高空65通道为主,地表按0.25弱权)
                _sl = spherical_spectral_loss if args.spec_mode == "sphere" else spectral_loss
                loss = loss + _sl(su.float(), tu, args.spec) + _sl(ss.float(), ts, args.spec * 0.25)
            for _opt, _ in opts:
                _opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():                       # fp16:仅单AdamW路径
                scaler.scale(loss).backward()
                scaler.unscale_(opts[0][0])
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opts[0][0]); scaler.update()
            else:                                          # bf16/fp32:Muon+AdamW通用
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                for _opt, _ in opts:
                    _opt.step()
            ema_update(ema, student)
            run += loss.item(); c += 1
        vrel = evaluate(ema)
        print(f"[ep{ep+1}/{args.epochs}] lr={args.lr*lr_at(ep):.2e} train={run/max(1,c):.4f} val_match={vrel*100:.2f}% {(time.time()-t0):.0f}s", flush=True)
        if args.dtp and ch_l1_n > 0:   # DTP:按per-channel对教师L1的EMA,难通道(L1大)自动加权,clamp防过冲
            ch_l1 = ch_l1_acc / ch_l1_n
            ch_l1_ema = ch_l1.clone() if ch_l1_ema is None else 0.9 * ch_l1_ema + 0.1 * ch_l1
            dyn_w = (ch_l1_ema / ch_l1_ema.mean()).clamp(0.3, 3.0)
            dyn_w = dyn_w / dyn_w.mean()
            print(f"  [dtp] dyn_w[{dyn_w.min():.2f},{dyn_w.max():.2f}] 最重通道#{dyn_w.argmax().item()}", flush=True)
        _cfg = {"embed": args.embed, "depths": list(depths), "heads": list(heads),
                "patch": list(patch), "global_mode": args.gm, "mlp_ratio": args.mlp_ratio, "residual": args.residual}
        if vrel > best:
            best = vrel
            torch.save({"model_state_dict": ema.state_dict(), "cfg": _cfg, "val_match": vrel, "epoch": ep + 1}, args.save)
            print(f"  [saved best] val_match={vrel*100:.2f}%", flush=True)
        # 每轮存断点(防DCU中断,重开同命令自动检测.ckpt从断点续训)
        torch.save({"model_state_dict": ema.state_dict(), "cfg": _cfg, "epoch": ep + 1, "best": best, "val_match": vrel}, ckpt_path)
    print(f"[done] best val_match={best*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
