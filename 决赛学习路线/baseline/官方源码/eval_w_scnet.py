# -*- coding: utf-8 -*-
"""SCNet 真实 W 评测(RMSE/ACC):教师 vs 学生,在 era5_real 测试年(2000),走 ERA5Datapipe(99->69过滤+训练同款归一化)。
反归一化后比真值,climatology=global_means。用法(pangu_weather/ 下):
  python eval_w_scnet.py --students /public/home/xdzs2026_c296/student_scnet_fast.pth
"""
import os, sys, json, argparse
import numpy as np, torch
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
import maxvit3d_student as M
from maxvit3d_student import MaxVit3DStudent
dev = 0
CKPT = "data/checkpoints/model_bak.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", default="/public/home/xdzs2026_c296/student_scnet_fast.pth")
    args = ap.parse_args()
    M.set_sdpa(False)
    cur = os.getcwd(); sys.path.append(cur)
    cfgp = os.path.join(cur, "conf/config.yaml")
    cfg = YParams(cfgp, "model"); cfg_data = YParams(cfgp, "datapipe")
    ds = cfg_data.dataset
    meta = json.load(open(os.path.join(ds.data_dir, "metadata.json")))
    allvars = meta["variables"]
    idx = [allvars.index(v) for v in ds.channels]  # 99->69
    mu = np.load(os.path.join(ds.stats_dir, "global_means.npy")).astype(np.float32)[:, idx]  # (1,69,1,1)
    sd = np.load(os.path.join(ds.stats_dir, "global_stds.npy")).astype(np.float32)[:, idx]
    mu_t = torch.tensor(mu, device=dev); sd_t = torch.tensor(sd, device=dev); clim = mu[0]
    sdir = ds.static_dir
    land = np.load(os.path.join(sdir, "land_mask.npy")).astype(np.float32)
    soil = np.load(os.path.join(sdir, "soil_type.npy")).astype(np.float32)
    topo = np.load(os.path.join(sdir, "topography.npy")).astype(np.float32); topo = (topo - topo.mean()) / (topo.std() + 1e-6)
    masks = torch.tensor(np.stack([land, soil, topo], 0), device=dev).unsqueeze(0)
    dp = ERA5Datapipe(params=cfg_data, distributed=False)
    test_loader = dp.test_dataloader()

    @torch.no_grad()
    def evalm(model):
        C = 69; rmse = np.zeros(C); num = np.zeros(C); p2 = np.zeros(C); l2 = np.zeros(C); n = 0
        for data in test_loader:
            invar, outvar = data[0], data[1]
            isf = invar[:, :4].to(dev, torch.float32); iua = invar[:, 4:].to(dev, torch.float32)
            x = torch.cat([isf, masks, iua], 1)
            ss, su = model(x); su = su.reshape(1, 65, x.shape[2], x.shape[3])
            pred = torch.cat([ss.float(), su.float()], 1)            # (1,69,H,W) normalized
            pred_phys = (pred * sd_t + mu_t)[0].cpu().numpy()
            gt_phys = (outvar.to(dev, torch.float32) * sd_t + mu_t)[0].cpu().numpy()
            rmse += np.sqrt(((gt_phys - pred_phys) ** 2).mean(axis=(1, 2)))
            pa = pred_phys - clim; la = gt_phys - clim
            num += (pa * la).sum(axis=(1, 2)); p2 += (pa ** 2).sum(axis=(1, 2)); l2 += (la ** 2).sum(axis=(1, 2))
            n += 1
        return rmse.mean() / max(1, n) * n / max(1, n), rmse.mean() / max(1, n), (num / (np.sqrt(p2 * l2) + 1e-8)).mean(), n
    # 简化:返回 avg RMSE, avg ACC, n
    @torch.no_grad()
    def W(model):
        C = 69; rmse = np.zeros(C); num = np.zeros(C); p2 = np.zeros(C); l2 = np.zeros(C); n = 0
        for data in test_loader:
            invar, outvar = data[0], data[1]
            isf = invar[:, :4].to(dev, torch.float32); iua = invar[:, 4:].to(dev, torch.float32)
            x = torch.cat([isf, masks, iua], 1)
            ss, su = model(x); su = su.reshape(1, 65, x.shape[2], x.shape[3])
            pred_phys = (torch.cat([ss.float(), su.float()], 1) * sd_t + mu_t)[0].cpu().numpy()
            gt_phys = (outvar.to(dev, torch.float32) * sd_t + mu_t)[0].cpu().numpy()
            rmse += np.sqrt(((gt_phys - pred_phys) ** 2).mean(axis=(1, 2)))
            pa = pred_phys - clim; la = gt_phys - clim
            num += (pa * la).sum(axis=(1, 2)); p2 += (pa ** 2).sum(axis=(1, 2)); l2 += (la ** 2).sum(axis=(1, 2)); n += 1
        return rmse.mean() / max(1, n), (num / (np.sqrt(p2 * l2) + 1e-8)).mean(), n

    teacher = Pangu(img_size=cfg_data.dataset.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads, window_size=cfg.window_size).to(dev).eval()
    ckt = torch.load(CKPT, map_location=f"cuda:{dev}", weights_only=False)
    teacher.load_state_dict(ckt["model_state_dict"] if "model_state_dict" in ckt else ckt)
    r, a, n = W(teacher)
    print(f"[test] era5_real 2000 pairs={n}", flush=True)
    print(f"[教师 Pangu 64.22M]  RMSE={r:.4f}  ACC={a:.4f}", flush=True)
    del teacher; torch.cuda.empty_cache()

    for sp in args.students.split(","):
        sp = sp.strip()
        ck = torch.load(sp, map_location=f"cuda:{dev}", weights_only=False)
        c = ck["cfg"]; sdk = ck["model_state_dict"]
        patch = tuple(c.get("patch", [2, 4, 4]))
        gm = "spectral" if any("mix." in k for k in sdk) else c.get("global_mode", "grid")
        st = MaxVit3DStudent(patch_size=patch, embed_dim=c["embed"],
                             depths=tuple(c["depths"]), num_heads=tuple(c["heads"]), global_mode=gm).to(dev).eval()
        st.load_state_dict(sdk)
        p = sum(t.numel() for t in st.parameters()) / 1e6
        r, a, n = W(st)
        print(f"[学生 {sp.split('/')[-1]} {p:.2f}M patch{patch} {gm}]  RMSE={r:.4f}  ACC={a:.4f}", flush=True)
        del st; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
