# -*- coding: utf-8 -*-
"""真值监督蒸馏(破天花板):连续 ERA5(era5_cont 相邻6h=t->t+6h真值对)。
温度通道(2m_t + temperature_1000/925/850)loss = α·教师 + (1-α)·真值,其余通道纯教师蒸馏。
纯蒸馏天花板=教师(W公式 min((教师RMSE/学生RMSE)²,1) 学生≤教师永远<满分);真值监督让温度直降 var → 可超教师。
可选物理头(--phys 1):在72ch输入后追加 N 个 0参数物理特征通道(太阳cos天顶角+时钟+sdor),patchembed2d Variables 7->7+N。

数据:era5_cont/data/{year}/{YYYYMMDDHH}.h5 (fields=69ch,顺序=CONFIG69,相邻文件差6h)。
用法(abc66, onescience311):
  PBASE=/data2/pangu/era5_cont PSTATS=/data2/pangu/xiandao2026-AI4S/onedatasets/ERA5_test \\
  PMETA=/data2/pangu/era5_seed/assets/era5_real_meta.json \\
  PCKPT=/data2/pangu/xiandao2026-AI4S/pangu_weather/data/checkpoints/model_bak.pth \\
  python distill_truth.py --years 1980 --epochs 30 --truth-alpha 0.5 --amp bf16 --resume <ratio2_dtp.pth> --save out.pth
"""
import os, glob, time, math, argparse, random
from datetime import datetime
import numpy as np, h5py, torch
import torch.nn.functional as F
from onescience.models.pangu import Pangu
from maxvit3d_student import MaxVit3DStudent
import maxvit3d_student as _M

CKPT = os.environ.get("PCKPT", "/data2/pangu/xiandao2026-AI4S/pangu_weather/data/checkpoints/model_bak.pth")
STATS = os.environ.get("PSTATS", "/data2/pangu/xiandao2026-AI4S/onedatasets/ERA5_test")
META99 = os.environ.get("PMETA", "/data2/pangu/era5_seed/assets/era5_real_meta.json")
BASE = os.environ.get("PBASE", "/data2/pangu/era5_cont")

# 救温度权重(与 SAVETEMP 一致:温度通道加重)
WEIGHTS = [1.50, 0.77, 0.66, 5.00] + [4.0,3.5,3.0]+[3.0]*10 + [0.6]*13 + [4.0,3.5,3.0,2.5,2.5,2.5,2.0,1.5,1.5,1.0,1.0,0.8,0.6] + [0.77]*13 + [0.54]*13
_LEV = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CONFIG69 = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"]
for _v in ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]:
    CONFIG69 += [f"{_v}_{_l}" for _l in _LEV]

# 温度通道索引:surf 中 2m_t = idx3;upper(65=5x13)中 temperature 是第3组(idx2)
T_SURF_IDX = [3]
T_UPPER_NEAR = [26, 27, 28]              # temperature_1000/925/850(近地面3层)
T_UPPER_FULL = list(range(26, 39))       # temperature 全13层(1000..50hPa)in 65-flat
T_UPPER_IDX = T_UPPER_NEAR               # 默认近地面;--full-temp 时改为全13层(main里设)


class DiskList:
    """落盘缓存的样本列表:存 .pt 路径,迭代时按需从磁盘读回 (x,ts,tu,trs,tru) 元组(全fp16)。
    支持 random.shuffle(就地洗路径)、len、迭代。规避 RAM 上限→可上全量数据。"""
    def __init__(self, paths):
        self.paths = list(paths)
    def __len__(self):
        return len(self.paths)
    def __iter__(self):
        for p in self.paths:
            yield torch.load(p, map_location="cpu", weights_only=False)
    def __getitem__(self, i):       # 供 random.shuffle 用
        return self.paths[i]
    def __setitem__(self, i, v):
        self.paths[i] = v


def wl1(x, y, w, lw):
    return lw * (F.l1_loss(x, y, reduction="none") * w).mean()


_HF_MASK = {}
def hf_spec_loss(pred, tgt, kmin, multiscale=False):
    """高频谱损失(治patch16低通过平滑):预测与真值各通道做rfft2,惩罚径向波数>kmin的高频幅度差。
    multiscale=True:中→高频渐进加权(波数越大权重越大,patch16杀得越狠的小尺度补得越多),覆盖中频15+;
    multiscale=False:二值mask(>kmin)。fp32计算+小权重+warmup→避谱vsRMSE梯度冲突。"""
    p = pred.float(); t = tgt.float()
    H, W = p.shape[-2], p.shape[-1]
    key = (H, W, kmin, multiscale, p.device)
    mask = _HF_MASK.get(key)
    if mask is None:
        fy = torch.fft.fftfreq(H, device=p.device).abs().view(H, 1) * H      # 行波数
        fx = torch.fft.rfftfreq(W, device=p.device).abs().view(1, -1) * W    # 列波数
        r = torch.sqrt(fy * fy + fx * fx)                                    # 径向波数
        if multiscale:
            # 多尺度:中频(kmin/2)起渐进加权,r越大权重越大(归一到[0,1]),覆盖中→高频
            k0 = kmin / 2.0; kmax = (H / 2.0)
            mask = ((r - k0) / (kmax - k0)).clamp(0.0, 1.0)                  # k0以下0,kmax满1,线性增
        else:
            mask = (r > kmin).float()                                        # 只留高频(二值)
        _HF_MASK[key] = mask
    pf = torch.fft.rfft2(p, norm="ortho"); tf = torch.fft.rfft2(t, norm="ortho")
    dmag = (pf.abs() - tf.abs()).abs() * mask
    return dmag.mean()


@torch.no_grad()
def ema_update(ema, model, d=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def build_pairs(years):
    """连续文件相邻配对:(files[i], files[i+1]) = (输入t, 真值t+6h),仅当文件名时间戳恰好差6h。"""
    pairs = []
    for y in years:
        fs = sorted(glob.glob(f"{BASE}/data/{y}/*.h5"))
        for i in range(len(fs) - 1):
            t0 = datetime.strptime(os.path.basename(fs[i])[:10], "%Y%m%d%H")
            t1 = datetime.strptime(os.path.basename(fs[i + 1])[:10], "%Y%m%d%H")
            if (t1 - t0).total_seconds() == 6 * 3600:
                pairs.append((fs[i], fs[i + 1]))
    return pairs


def read_field(f, mu, sd, sel69):
    with h5py.File(f, "r") as h:
        fld = h["fields"][:].astype(np.float32)
    if fld.shape[0] != 69:
        fld = fld[sel69]
    return (fld - mu) / sd  # (69,721,1440) 归一化


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=96)
    ap.add_argument("--depths", type=str, default="2,4,2")
    ap.add_argument("--heads", type=str, default="6,12,6")
    ap.add_argument("--patch", type=str, default="2,16,16")
    ap.add_argument("--years", type=str, default="1980")
    ap.add_argument("--val-frac", dest="vf", type=float, default=0.1, help="末尾比例做验证")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay(加大如0.15-0.2压权重→间接压激活,治fp16激活溢出)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sdpa", type=int, default=1)
    ap.add_argument("--amp", type=str, default="bf16")
    ap.add_argument("--max-cache", dest="maxc", type=int, default=600)
    ap.add_argument("--global-mode", dest="gm", type=str, default="grid")
    ap.add_argument("--mlp-ratio", dest="mlp_ratio", type=float, default=2.0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--save", type=str, default="/data2/pangu/student_truth.pth")
    ap.add_argument("--opt", type=str, default="muon", choices=["adamw", "muon"])
    ap.add_argument("--muon-lr", dest="muon_lr", type=float, default=0.02)
    ap.add_argument("--muon-momentum", dest="muon_momentum", type=float, default=0.99)
    ap.add_argument("--dtp", type=int, default=1)
    # 真值监督
    ap.add_argument("--truth-alpha", dest="alpha", type=float, default=0.5,
                    help="温度通道 loss = α·教师 + (1-α)·真值;α=1纯蒸馏 α=0纯真值")
    # 物理头
    ap.add_argument("--phys", type=int, default=0, help="1=输入追加物理特征通道(太阳cos天顶角+时钟4+sdor),Variables 7->7+N")
    ap.add_argument("--full-temp", dest="full_temp", type=int, default=0,
                    help="1=真值监督覆盖全部13个温度层(治高空温度漂移),0=仅近地面3层")
    ap.add_argument("--protect", type=float, default=1.0,
                    help="风/位势保护倍数:对 u/v(各层+10m)+geopotential+msl 的蒸馏权重×此倍数,"
                         "强化它们对教师的锚定,防温度真值训练通过共享backbone把它们带歪(>1 如1.5/2.0)")
    ap.add_argument("--disk-cache", dest="disk_cache", type=str, default="",
                    help="落盘缓存目录:教师目标+真值落盘(每对一个.pt),训练按需读→规避RAM上限上全量。空=RAM缓存")
    # 第1搏:针对性回收低层风/湿(W归因:劣化集中低层u/v+u10/v10+q,非温度)
    ap.add_argument("--chan-reweight", dest="chanrw", type=float, default=0.0,
                    help=">0开启通道重加权:低层(1000/925/850)u/v+u10/v10+q上调此倍数(如2.5),已追平的t/z降0.8×")
    ap.add_argument("--region-weight", dest="regionw", type=float, default=0.0,
                    help=">0开启区域加权:陆地+高地形+极地(|lat|>60)格点×此倍数(如1.5-2),治陆/极地劣化")
    ap.add_argument("--tz-scale", dest="tzscale", type=float, default=0.8,
                    help="chan-reweight时t/z(已追平)的降权(0.8省梯度但温度退化;1.0不降温度不退化)")
    ap.add_argument("--chan-renorm", dest="chan_renorm", type=int, default=0,
                    help="1=chan-reweight后归一化总权重不变(只重分配emphasis不抬高总梯度→不放大激活,治fp16溢出)")
    ap.add_argument("--truth-windq", dest="truth_windq", type=int, default=0,
                    help="1=真值监督掺低层u/v/q(回收过平滑高频),配合温度真值;0=只温度真值")
    ap.add_argument("--truth-full15", dest="truth_full15", type=int, default=0,
                    help="1=全15指标真值监督(冲W满分):surf全4+upper全65都掺真值α教师+(1-α)真值,直接学ERA5超3年前教师")
    ap.add_argument("--alpha-f15", dest="alpha_f15", type=float, default=0.5,
                    help="全15指标真值监督的α(教师占比),(1-α)真值占比;0.5起,越小越靠真值")
    # 第2搏:高频谱损失(治patch16低通过平滑,逼还原波数>30中小尺度)
    ap.add_argument("--hf-spec", dest="hfspec", type=float, default=0.0,
                    help=">0开启高频谱损失:预测vs真值rfft,惩罚波数>30高频差(权重,建议0.1起)")
    ap.add_argument("--hf-warmup", dest="hf_warmup", type=int, default=5,
                    help="高频谱损失warmup轮数(慢加权重避免谱vsRMSE梯度冲突)")
    ap.add_argument("--hf-kmin", dest="hf_kmin", type=int, default=30, help="高频谱损失起始波数(只补>此波数)")
    ap.add_argument("--hf-multiscale", dest="hf_ms", type=int, default=0, help="1=多尺度谱损失(中→高频渐进加权,覆盖kmin/2起)")
    args = ap.parse_args()
    global T_UPPER_IDX
    if args.full_temp:
        T_UPPER_IDX = T_UPPER_FULL
        print(f"[truth] 全温度层真值监督(13层 temperature_1000..50)", flush=True)
    else:
        T_UPPER_IDX = T_UPPER_NEAR
        print(f"[truth] 近地面温度真值监督(3层 temperature_1000/925/850)", flush=True)
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
    topo_raw = np.load(f"{STATS}/static/topography.npy").astype(np.float32)
    topo = (topo_raw - topo_raw.mean()) / (topo_raw.std() + 1e-6)
    masks = np.stack([land, soil, topo], 0)
    m99 = json.load(open(META99))["variables"]
    sel69 = np.array([m99.index(v) for v in CONFIG69])
    if mu.shape[0] != 69:
        mu = mu[sel69]; sd = sd[sel69]

    # 物理头特征(0参数,几何物理量)
    phys_static = None; n_phys = 0
    if args.phys:
        from phys_features import solar_zenith_cos, clock_feats, sdor_field
        lat2d = np.repeat(np.linspace(90, -90, 721)[:, None], 1440, 1).astype(np.float32)
        lon2d = np.repeat(np.linspace(0, 360, 1440, endpoint=False)[None, :], 721, 0).astype(np.float32)
        sdor = sdor_field(topo_raw)  # 次网格地形std
        sdor = (sdor - sdor.mean()) / (sdor.std() + 1e-6)
        n_phys = 1 + 4 + 1  # solar(1) + clock(4 broadcast) + sdor(1)
        phys_static = dict(lat2d=lat2d, lon2d=lon2d, sdor=sdor.astype(np.float32))
        print(f"[phys] 物理头开启,追加 {n_phys} 通道(solar1+clock4+sdor1),Variables 7->{7+n_phys}", flush=True)

    def make_phys(dtstr):
        """生成 (n_phys,721,1440) 物理特征(归一化空间已在内部处理)。"""
        dt = datetime.strptime(dtstr, "%Y%m%d%H")
        solar = solar_zenith_cos(dt, phys_static["lat2d"], phys_static["lon2d"])  # 0..1
        ck = clock_feats(dt)  # 4 标量
        clk = np.stack([np.full((721, 1440), v, np.float32) for v in ck], 0)  # (4,721,1440)
        out = np.concatenate([solar[None], clk, phys_static["sdor"][None]], 0)  # (6,721,1440)
        return out.astype(np.float32)

    sw = torch.tensor(WEIGHTS[:4], device=dev).view(1, 4, 1, 1)
    pw = torch.tensor(WEIGHTS[4:], device=dev).view(1, 65, 1, 1)
    if args.protect != 1.0:
        # 风/位势保护:加强这些通道的教师蒸馏权重(它们已贴教师,锚牢别被温度真值带歪)
        # surface: msl(0)/u10(1)/v10(2);upper: geo[0:13]/u[39:52]/v[52:65]
        sw[0, 0] *= args.protect; sw[0, 1] *= args.protect; sw[0, 2] *= args.protect
        pw[0, 0:13] *= args.protect          # geopotential 13层
        pw[0, 39:65] *= args.protect         # u 13层 + v 13层
        print(f"[protect] 风/位势蒸馏权重×{args.protect}(msl/u10/v10 + geo + u/v 全层),防被温度真值带歪", flush=True)
    if args.chanrw > 0:
        # 通道重加权(W归因:劣化集中低层风u/v+u10/v10+q,非温度/位势)
        _sw0_sum = float(sw.sum()); _pw0_sum = float(pw.sum())   # reweight前总权重(归一化用)
        # surf: u10=1,v10=2 上调;2m_t=3 降0.8(已追平)。msl=0 不动(陆地靠区域加权)
        sw[0, 1] *= args.chanrw; sw[0, 2] *= args.chanrw; sw[0, 3] *= args.tzscale
        # upper(65): q[13:26]全层上调(尤其300-1000); u[39:52]+v[52:65] 低层(0,1,2=1000/925/850)上调
        pw[0, 13:26] *= args.chanrw                                  # 比湿q 全层
        for base in (39, 52):                                        # u组/v组
            pw[0, base + 0] *= args.chanrw; pw[0, base + 1] *= args.chanrw; pw[0, base + 2] *= args.chanrw  # 1000/925/850
        pw[0, 0:13] *= args.tzscale                                  # geopotential 降(已追平)
        pw[0, 26:39] *= args.tzscale                                 # temperature 降(已追平)
        if args.chan_renorm:
            # 归一化:重加权后把sw/pw总均值拉回原值→只重分配emphasis,不抬高总梯度幅度→不放大激活(治fp16溢出)
            tot_before = float(_sw0_sum + _pw0_sum)
            tot_after = float(sw.sum() + pw.sum())
            scale = tot_before / (tot_after + 1e-8)
            sw *= scale; pw *= scale
            print(f"[chanrw] 低层u/v/q ×{args.chanrw} t/z×{args.tzscale} + 归一化(总权重不变×{scale:.3f}防激活溢出)", flush=True)
        else:
            print(f"[chanrw] 低层u/v+u10/v10+q ×{args.chanrw},t/z ×{args.tzscale}(省梯度给劣化通道)", flush=True)
    _lat = torch.cos(torch.deg2rad(torch.linspace(90, -90, 721, device=dev)))
    coslat = (_lat / _lat.mean()).view(1, 1, 721, 1)
    # 区域加权(W归因:陆地msl1.13vs海1.005,极地差热带3×,高地形劣化)
    regionw = None
    if args.regionw > 0:
        rw = np.ones((721, 1440), np.float32)
        rw = rw + (land > 0.5) * (args.regionw - 1.0)               # 陆地×regionw
        topo_n = (topo_raw - topo_raw.mean()) / (topo_raw.std() + 1e-6)
        rw = rw + (topo_n > 1.0) * (args.regionw - 1.0)             # 高地形(>1std)再叠加
        lat1d = np.abs(np.linspace(90, -90, 721))
        polar = (lat1d > 60)[:, None]                                # |lat|>60极地
        rw = rw + polar * (args.regionw - 1.0)
        regionw = torch.from_numpy(rw).to(dev).view(1, 1, 721, 1440)
        regionw = regionw / regionw.mean()                          # 归一化均值1(不改整体尺度)
        print(f"[region] 陆地+高地形+极地 ×{args.regionw}(归一化均值1)", flush=True)
    # 反归一化std(温度通道,用于真值RMSE物理K):surf全4 + upper温度3层
    sd_s = torch.from_numpy(sd[:4]).to(dev).view(1, 4, 1, 1)
    sd_ut = torch.from_numpy(sd[4:][T_UPPER_IDX]).to(dev).view(1, len(T_UPPER_IDX), 1, 1)

    # 温度通道掩膜(在 sw/pw 上对真值监督做差异化:真值项只作用温度通道)
    tmask_s = torch.zeros(1, 4, 1, 1, device=dev); tmask_s[0, T_SURF_IDX] = 1.0
    tmask_u = torch.zeros(1, 65, 1, 1, device=dev); tmask_u[0, T_UPPER_IDX] = 1.0

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    pairs = build_pairs(years)
    print(f"[data] 连续6h对 {len(pairs)} (years={years})", flush=True)
    random.seed(0); random.shuffle(pairs)
    pairs = pairs[:args.maxc]
    nv = max(1, int(len(pairs) * args.vf))
    va_pairs = pairs[:nv]; tr_pairs = pairs[nv:]
    print(f"[split] train={len(tr_pairs)} val={len(va_pairs)}", flush=True)

    teacher = Pangu(img_size=(721, 1440)).to(dev).eval()
    ck = torch.load(CKPT, map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    for p in teacher.parameters():
        p.requires_grad_(False)
    if args.amp == "fp16":
        teacher.half()
    print("[teacher] loaded", flush=True)

    def teach(x):
        with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
            ts, tu = teacher(x)
        return ts.float(), tu.reshape(x.shape[0], 65, x.shape[2], x.shape[3]).float()

    def cache(plist, tag=""):
        """每对存 (x_in[72(+phys)], teacher_s[4], teacher_u[65], truth_s[4], truth_u[3温度层])，全 fp16。
        disk_cache 非空→落盘成 .pt 返回 DiskList(规避RAM上限);否则 RAM 列表(旧行为)。"""
        c = []; t0 = time.time(); skipped = 0
        disk = bool(args.disk_cache)
        if disk:
            os.makedirs(os.path.join(args.disk_cache, tag), exist_ok=True)
        with torch.no_grad():
            for i, (fi, fj) in enumerate(plist):
                if disk:
                    pth = os.path.join(args.disk_cache, tag, f"{i:05d}.pt")
                    if os.path.exists(pth):     # 复用已落盘.pt(换lr/超参重训省重cache;同truth_full配置才可复用)
                        c.append(pth); continue
                try:
                    f_in = read_field(fi, mu, sd, sel69)   # (69,...) 归一化输入t
                    f_tr = read_field(fj, mu, sd, sel69)   # (69,...) 归一化真值t+6h
                except Exception:
                    skipped += 1; continue
                # 标准布局 surf4+mask3+upper65;物理头版插在 mask 后:surf4+mask3+phys_n+upper65
                if args.phys:
                    p = make_phys(os.path.basename(fi)[:10])         # (n_phys,...) 用输入t的时间
                    xin = np.concatenate([f_in[:4], masks, p, f_in[4:]], 0)  # (72+n_phys,...)
                    # 教师吃标准72(去掉物理通道):surf4+mask3+upper65
                    xt72 = np.concatenate([f_in[:4], masks, f_in[4:]], 0)
                else:
                    xin = np.concatenate([f_in[:4], masks, f_in[4:]], 0)  # (72,...)
                    xt72 = xin
                x = torch.from_numpy(xin).unsqueeze(0).to(dev)
                xt = torch.from_numpy(xt72).unsqueeze(0).to(dev)
                ts, tu = teach(xt)
                # 真值:默认只存温度层省内存;--truth-windq/--hf-spec 需全65层upper真值(回收风/湿高频)
                truth_s = torch.from_numpy(f_tr[:4]).unsqueeze(0)            # (1,4,...)
                f_up = f_tr[4:]                                              # (65,...)
                if args.truth_windq or args.hfspec > 0 or args.truth_full15:
                    truth_ut = torch.from_numpy(f_up).unsqueeze(0)          # (1,65,...) 全upper真值
                else:
                    truth_ut = torch.from_numpy(f_up[T_UPPER_IDX]).unsqueeze(0)  # (1,温度层,...)
                item = (x.half().cpu(), ts.half().cpu(), tu.half().cpu(),
                        truth_s.half(), truth_ut.half())
                if disk:
                    pth = os.path.join(args.disk_cache, tag, f"{i:05d}.pt")
                    torch.save(item, pth); c.append(pth)
                else:
                    c.append(item)
                if (i + 1) % 50 == 0:
                    print(f"  [cache] {i+1}/{len(plist)} {(time.time()-t0):.0f}s", flush=True)
        if skipped:
            print(f"  [cache] 跳过 {skipped} 坏文件", flush=True)
        return (DiskList(c) if disk else c), time.time() - t0

    tr, tt = cache(tr_pairs, "train")
    va, _ = cache(va_pairs, "val")
    print(f"[cache] train={len(tr)} val={len(va)} 教师预算{tt:.0f}s", flush=True)
    del teacher; torch.cuda.empty_cache()

    in_chans2d = 7 + n_phys
    student = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths, num_heads=heads,
                              use_checkpoint=False, global_mode=args.gm, mlp_ratio=args.mlp_ratio).to(dev)
    ema = MaxVit3DStudent(patch_size=patch, embed_dim=args.embed, depths=depths, num_heads=heads,
                          global_mode=args.gm, mlp_ratio=args.mlp_ratio).to(dev).eval()
    # 物理头:扩 patchembed2d 输入通道 7->7+n_phys(新通道权重0初始化=不破坏resume的原7通道)+ patch forward 切片
    if args.phys:
        _expand_embed2d(student, in_chans2d)
        _expand_embed2d(ema, in_chans2d)
        _patch_forward_phys(student, in_chans2d)
        _patch_forward_phys(ema, in_chans2d)

    start_ep = 0; best = -1e9   # sel改为对真值负RMSE,初始用-1e9(原-1.0是val_match口径,新口径下永不触发saved best)
    ckpt_path = args.save + ".ckpt"
    resume_from = args.resume if (args.resume and os.path.exists(args.resume)) else (ckpt_path if os.path.exists(ckpt_path) else "")
    if resume_from:
        rk = torch.load(resume_from, map_location=f"cuda:{dev}", weights_only=False)
        rsd = rk["model_state_dict"] if "model_state_dict" in rk else rk
        if args.phys:  # resume 的是7通道stem,先把新通道补0再灌(保留原7通道权重)
            rsd = _adapt_resume_phys(rsd, student.state_dict())
        miss = student.load_state_dict(rsd, strict=False)
        if not args.resume == ckpt_path:  # 指定外部resume(换数据/破天花板)→从ep0重新跑调度,best清零
            start_ep = 0; best = -1e9  # 对真值sel口径(负RMSE),外部resume best清零用-1e9
        else:
            start_ep = rk.get("epoch", 0); best = rk.get("best", rk.get("val_match", -1.0))
        print(f"[resume] 从 {resume_from} 续训(missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)})", flush=True)
    ema.load_state_dict(student.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ps = sum(p.numel() for p in student.parameters())
    print(f"[model] student={ps/1e6:.2f}M (in2d={in_chans2d}ch, alpha={args.alpha}, phys={args.phys})", flush=True)

    if args.opt == "muon":
        from muon import Muon
        muon_p, adamw_p = [], []
        for n, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and not any(k in n for k in ("patchembed", "recovery", "embed", "logit_scale")):
                muon_p.append(p)
            else:
                adamw_p.append(p)
        opt_m = Muon(muon_p, lr=args.muon_lr, momentum=args.muon_momentum)
        opt_a = torch.optim.AdamW(adamw_p, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
        opts = [(opt_m, args.muon_lr), (opt_a, args.lr)]
        print(f"[opt] Muon {len(muon_p)} + AdamW {len(adamw_p)}", flush=True)
    else:
        opt_a = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
        opts = [(opt_a, args.lr)]
        print(f"[opt] AdamW", flush=True)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    truth_full = bool(args.truth_windq or args.hfspec > 0 or args.truth_full15)   # tru 是否为全65层upper真值
    def tru_temp(tru):  # 从 tru 取温度层(全65→T_UPPER_IDX切片;温度版→原样)
        return tru[:, T_UPPER_IDX] if truth_full else tru

    @torch.no_grad()
    def evaluate(model):
        """口径:对【真值t+6h】算温度通道RMSE(物理意义),同时报 val_match(对教师)。"""
        model.eval()
        se_t2m = 0.0; se_tup = torch.zeros(len(T_UPPER_IDX), device=dev); n = 0; rel = 0.0
        # 教师对真值的温度RMSE(天花板参照),只算一次
        for x, ts, tu, trs, tru in va:
            x = x.float().to(dev)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                ss, su = model(x); su = su.reshape(tu.shape)
            ss, su = ss.float(), su.float()
            trs = trs.float().to(dev); tru = tru_temp(tru.float().to(dev))   # 温度层真值
            tsd = ts.float().to(dev); tud = tu.float().to(dev)
            # 真值RMSE(物理K):反归一化。2m_t=surf idx3;温度层=su[:,T_UPPER_IDX] vs tru
            ds = (ss[:, 3] - trs[:, 3]) * sd_s[0, 3]
            su_t = su[:, T_UPPER_IDX]; du = (su_t - tru) * sd_ut
            se_t2m += (ds ** 2).mean().item()
            se_tup += torch.stack([(du[:, j] ** 2).mean() for j in range(len(T_UPPER_IDX))])
            num = (ss - tsd).abs().mean() + (su - tud).abs().mean()
            den = tsd.abs().mean() + tud.abs().mean() + 1e-6
            rel += (1 - num / den).item(); n += 1
        rmse_t2m = math.sqrt(se_t2m / max(1, n))
        rmse_tup = (se_tup / max(1, n)).sqrt().tolist()
        return rel / max(1, n), rmse_t2m, rmse_tup

    @torch.no_grad()
    def teacher_truth_rmse():
        """教师对真值的温度RMSE(天花板:学生破此即超教师)。"""
        se2 = 0.0; seu = torch.zeros(len(T_UPPER_IDX), device=dev); n = 0
        for x, ts, tu, trs, tru in va:
            ts = ts.float().to(dev); tu = tu.float().to(dev)
            trs = trs.float().to(dev); tru = tru_temp(tru.float().to(dev))
            ds = (ts[:, 3] - trs[:, 3]) * sd_s[0, 3]
            tu_t = tu[:, T_UPPER_IDX]; du = (tu_t - tru) * sd_ut
            se2 += (ds ** 2).mean().item()
            seu += torch.stack([(du[:, j] ** 2).mean() for j in range(len(T_UPPER_IDX))]); n += 1
        return math.sqrt(se2 / max(1, n)), (seu / max(1, n)).sqrt().tolist()

    te_t2m, te_tup = teacher_truth_rmse()
    print(f"[teacher天花板] 对真值 2m_t RMSE={te_t2m:.3f}K  t1000/925/850={[round(v,3) for v in te_tup]}", flush=True)

    dyn_w = torch.ones(69, device=dev); ch_l1_ema = None
    for ep in range(start_ep, args.epochs):
        for _opt, _base in opts:
            for g in _opt.param_groups:
                g["lr"] = _base * lr_at(ep)
        student.train(); t0 = time.time(); run = 0.0; c = 0
        ch_l1_acc = torch.zeros(69, device=dev); ch_l1_n = 0
        random.shuffle(tr)
        for x, ts, tu, trs, tru in tr:
            x = x.float().to(dev); ts = ts.float().to(dev); tu = tu.float().to(dev)
            trs = trs.float().to(dev); tru = tru.float().to(dev)
            rgn_roll = regionw
            if args.aug:
                s = random.randint(0, x.shape[-1] - 1)
                x = torch.roll(x, s, -1); ts = torch.roll(ts, s, -1); tu = torch.roll(tu, s, -1)
                trs = torch.roll(trs, s, -1); tru = torch.roll(tru, s, -1)
                if regionw is not None:   # 区域权重(含经度相关的陆地/地形)须同步roll,否则错位
                    rgn_roll = torch.roll(regionw, s, -1)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                ss, su = student(x); su = su.reshape(tu.shape)
            ssf, suf = ss.float(), su.float()
            rgn = rgn_roll if rgn_roll is not None else 1.0   # 区域加权(陆地+高地形+极地)
            wsd = sw * dyn_w[:4].view(1, 4, 1, 1) * coslat * rgn
            wud = pw * dyn_w[4:].view(1, 65, 1, 1) * coslat * rgn
            tru_t = tru_temp(tru)   # 温度层真值(tru可能是全65层)
            a = args.alpha
            if args.truth_full15:
                # 全15指标真值监督:全通道教师×α_f15(留(1-α)给真值项),不再用温度掩膜
                af = args.alpha_f15
                loss = wl1(ssf, ts, wsd * af, 0.25) + wl1(suf, tu, wud * af, 1.0)
            else:
                # 非温度通道:纯教师;温度通道:α教师+(1-α)真值
                te_w_s = wsd * (1 - tmask_s * (1 - a))
                te_w_u = wud * (1 - tmask_u * (1 - a))
                loss = wl1(ssf, ts, te_w_s, 0.25) + wl1(suf, tu, te_w_u, 1.0)
            # 温度真值项(权重×(1-α));trs全4ch(用2m_t),tru_t温度层。truth_full15时跳过(被全15覆盖避免重复)
            if (1 - a) > 0 and not args.truth_full15:
                w2m = wsd[:, 3:4] * (1 - a)
                loss = loss + 0.25 * (F.l1_loss(ssf[:, 3:4], trs[:, 3:4], reduction="none") * w2m).mean()
                wtu = wud[:, T_UPPER_IDX] * (1 - a)
                loss = loss + 1.0 * (F.l1_loss(suf[:, T_UPPER_IDX], tru_t, reduction="none") * wtu).mean()
            # 第1搏③:真值监督掺低层u/v/q(回收过平滑高频)。tru是全65层真值。truth_full15时跳过(被覆盖)
            if args.truth_windq and not args.truth_full15:
                a2 = args.alpha
                # 低层u/v(1000/925/850=组内0,1,2)+ q全层(13:26)+ u10/v10(surf 1,2)
                widx = [39, 40, 41, 52, 53, 54] + list(range(13, 26))   # u低3+v低3+q全13
                wq = wud[:, widx] * (1 - a2)
                loss = loss + 1.0 * (F.l1_loss(suf[:, widx], tru[:, widx], reduction="none") * wq).mean()
                w10 = wsd[:, 1:3] * (1 - a2)   # u10/v10
                loss = loss + 0.25 * (F.l1_loss(ssf[:, 1:3], trs[:, 1:3], reduction="none") * w10).mean()
            # 第3搏:全15指标真值监督(冲W满分,直接学ERA5真值超3年前教师)。tru全65层真值
            # 覆盖15指标对应全通道:surf全4(msl/10u/10v/2m_t)+ upper全65(geo/q/temp/u/v全层防高空漂移)
            if args.truth_full15:
                a3 = args.alpha_f15
                # surf全4(2m_t已在温度项,这里msl/10u/10v + 2m_t统一真值)
                wsf = wsd * (1 - a3)
                loss = loss + 0.25 * (F.l1_loss(ssf, trs, reduction="none") * wsf).mean()
                # upper全65层真值(geo/q/temp/u/v全层,full式防只监督近地面致高空漂移)
                wuf = wud * (1 - a3)
                loss = loss + 1.0 * (F.l1_loss(suf, tru, reduction="none") * wuf).mean()
            # 第2搏:高频谱损失(治patch16低通过平滑,逼还原波数>kmin中小尺度)
            if args.hfspec > 0:
                hw = args.hfspec * min(1.0, (ep - start_ep + 1) / max(1, args.hf_warmup))  # warmup慢加
                loss = loss + hw * hf_spec_loss(suf, tru, args.hf_kmin, bool(args.hf_ms))
                loss = loss + hw * hf_spec_loss(ssf, trs, args.hf_kmin, bool(args.hf_ms)) * 0.25
            if args.dtp:
                with torch.no_grad():
                    ch_l1_acc += torch.cat([(ssf - ts).abs().mean(dim=(0, 2, 3)),
                                            (suf - tu).abs().mean(dim=(0, 2, 3))]); ch_l1_n += 1
            for _opt, _ in opts:
                _opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opts[0][0])
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opts[0][0]); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                for _opt, _ in opts:
                    _opt.step()
            ema_update(ema, student)
            run += loss.item(); c += 1
        vrel, r2m, rup = evaluate(ema)
        print(f"[ep{ep+1}/{args.epochs}] lr={args.lr*lr_at(ep):.2e} train={run/max(1,c):.4f} "
              f"val_match={vrel*100:.2f}% | 学生对真值 2m_t={r2m:.3f}K(教师{te_t2m:.3f}) "
              f"t1000/925/850={[round(v,3) for v in rup]} {(time.time()-t0):.0f}s", flush=True)
        if args.dtp and ch_l1_n > 0:
            ch_l1 = ch_l1_acc / ch_l1_n
            ch_l1_ema = ch_l1.clone() if ch_l1_ema is None else 0.9 * ch_l1_ema + 0.1 * ch_l1
            dyn_w = (ch_l1_ema / ch_l1_ema.mean()).clamp(0.3, 3.0); dyn_w = dyn_w / dyn_w.mean()
        _cfg = {"embed": args.embed, "depths": list(depths), "heads": list(heads),
                "patch": list(patch), "global_mode": args.gm, "mlp_ratio": args.mlp_ratio,
                "residual": False, "phys": args.phys, "n_phys": n_phys}
        # 选优口径(冲W满分):对【真值】越好越优,而非 val_match(对教师)——后者会选最像教师的、非最超真值的。
        # 真值监督核心=超真值,故 sel = -(对真值温度总RMSE)。RMSE越小→sel越大→存为best。
        sel = -(r2m + (sum(rup) / max(1, len(rup))))  # 对真值:2m_t + 温度层平均RMSE,越小越优
        if sel > best:
            best = sel
            torch.save({"model_state_dict": ema.state_dict(), "cfg": _cfg, "val_match": vrel,
                        "rmse_t2m": r2m, "rmse_tup": rup, "epoch": ep + 1}, args.save)
            print(f"  [saved best] val_match={vrel*100:.2f}% 2m_t={r2m:.3f}K", flush=True)
        torch.save({"model_state_dict": ema.state_dict(), "cfg": _cfg, "epoch": ep + 1,
                    "best": best, "val_match": vrel}, ckpt_path)
    print(f"[done] best val_match={best*100:.2f}%", flush=True)


def _expand_embed2d(model, new_in):
    """把 patchembed2d 的输入通道从7扩到 new_in,新通道权重0初始化(不破坏已有7通道)。"""
    import torch.nn as nn
    pe = model.patchembed2d
    # 找到 conv proj(split=embedder.proj;unified=embedder.proj 或 proj)
    proj = None; parent = None; name = None
    for cand_parent, cand_name in [(pe, "proj"), (getattr(pe, "embedder", None), "proj")]:
        if cand_parent is not None and hasattr(cand_parent, cand_name):
            proj = getattr(cand_parent, cand_name); parent = cand_parent; name = cand_name; break
    if proj is None:
        raise RuntimeError("找不到 patchembed2d 的 conv proj")
    w = proj.weight.data  # (embed, 7, [1,]kh, kw) split=(E,7,16,16) unified=(E,7,1,16,16)
    old_in = w.shape[1]
    if old_in >= new_in:
        return
    extra = new_in - old_in
    pad = torch.zeros((w.shape[0], extra, *w.shape[2:]), dtype=w.dtype, device=w.device)
    new_w = torch.cat([w, pad], dim=1)
    new_conv = nn.Conv2d(new_in, proj.out_channels, kernel_size=proj.kernel_size,
                         stride=proj.stride, padding=proj.padding, bias=(proj.bias is not None)) \
        if isinstance(proj, nn.Conv2d) else \
        nn.Conv3d(new_in, proj.out_channels, kernel_size=proj.kernel_size,
                  stride=proj.stride, padding=proj.padding, bias=(proj.bias is not None))
    new_conv.weight.data = new_w
    if proj.bias is not None:
        new_conv.bias.data = proj.bias.data.clone()
    setattr(parent, name, new_conv.to(w.device))


def _patch_forward_phys(model, n_surf):
    """物理头版 forward:surface 取前 n_surf(=7+n_phys),upper 取其后(5x13)。其余与原 forward 一致。
    避免改全局 maxvit3d_student.py(其它实验复用)。"""
    import types, torch as _t
    def fwd(self, x):
        surface = x[:, :n_surf, :, :]                                # B, n_surf, 721, 1440
        upper = x[:, n_surf:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper = self.patchembed3d(upper)
        z = _t.concat([surface.unsqueeze(2), upper], dim=2)
        B, C, Pl, Lat, Lon = z.shape
        z = z.reshape(B, C, -1).transpose(1, 2)
        z = self._run(self.enc, z, self.res_full)
        skip = z
        z = self.downsample(z)
        z = self._run(self.mid, z, self.res_down)
        z = self.upsample(z)
        z = self._run(self.dec, z, self.res_full)
        output = _t.concat([z, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        out_surface = self.patchrecovery2d(output[:, :, 0, :, :])
        out_upper = self.patchrecovery3d(output[:, :, 1:, :, :])
        return out_surface, out_upper
    model.forward = types.MethodType(fwd, model)


def _adapt_resume_phys(rsd, model_sd):
    """resume 的7通道stem权重 → 适配到 7+n_phys:新通道补0。其余键直接用。"""
    out = {}
    for k, v in rsd.items():
        mk = model_sd.get(k)
        if mk is not None and tuple(mk.shape) != tuple(v.shape) and "patchembed2d" in k and v.ndim >= 2:
            # 通道维(dim=1)扩展
            if mk.shape[1] > v.shape[1]:
                pad = torch.zeros((v.shape[0], mk.shape[1] - v.shape[1], *v.shape[2:]), dtype=v.dtype, device=v.device)
                out[k] = torch.cat([v.to(v.device), pad], dim=1)
                continue
        out[k] = v
    return out


if __name__ == "__main__":
    main()
