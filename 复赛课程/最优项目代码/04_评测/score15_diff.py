# -*- coding: utf-8 -*-
"""逐通道 W 对比: W1 vs bo3b vs 教师, 找 truth-full15 哪些通道涨/跌(指导方向).
fp32 @ era5_real/2000. 用法: python score15_diff.py <w1.pth> [bo3b.pth]"""
import os, sys, json
os.chdir("/public/home/xdzs2026_c296/submit_pw"); sys.path.insert(0,".")
W1 = sys.argv[1]
BO = sys.argv[2] if len(sys.argv)>2 else "data/checkpoints/student.pth"
import numpy as np, torch
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
import maxvit3d_student as M; M.set_sdpa(False)
from maxvit3d_student import MaxVit3DStudent
dev=0
cfgd=YParams("conf/config.yaml","datapipe"); cfg=YParams("conf/config.yaml","model"); ds=cfgd.dataset
ds.data_dir="/public/home/xdzs2026_c296/era5_real"; ds.stats_dir=ds.data_dir+"/stats"; ds.static_dir=ds.data_dir+"/static"; ds.test_ratio=[2000]
meta=json.load(open(ds.data_dir+"/metadata.json")); idx=[meta["variables"].index(v) for v in ds.channels]
mu=np.load(ds.stats_dir+"/global_means.npy").astype(np.float32)[:,idx]; sd=np.load(ds.stats_dir+"/global_stds.npy").astype(np.float32)[:,idx]; clim=mu[0]
mu_t=torch.tensor(mu,device=dev); sd_t=torch.tensor(sd,device=dev)
land=np.load(ds.static_dir+"/land_mask.npy").astype(np.float32); soil=np.load(ds.static_dir+"/soil_type.npy").astype(np.float32)
topo=np.load(ds.static_dir+"/topography.npy").astype(np.float32); topo=(topo-topo.mean())/(topo.std()+1e-6)
masks=torch.tensor(np.stack([land,soil,topo],0),device=dev).unsqueeze(0)
OFFICIAL15=["mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","geopotential_850","geopotential_700","geopotential_500","specific_humidity_850","specific_humidity_700","specific_humidity_500","temperature_850","temperature_700","temperature_500","u_component_of_wind_500","v_component_of_wind_500"]
sel=[ds.channels.index(v) for v in OFFICIAL15]
dp=ERA5Datapipe(params=cfgd,distributed=False)
def remap_state(sdk,msd):
    if "patchembed2d.embedder.proj.weight" in sdk: return sdk
    out=dict(sdk)
    if "patchembed2d.proj.weight" in out: out["patchembed2d.embedder.proj.weight"]=out.pop("patchembed2d.proj.weight").reshape(msd["patchembed2d.embedder.proj.weight"].shape)
    if "patchembed2d.proj.bias" in out: out["patchembed2d.embedder.proj.bias"]=out.pop("patchembed2d.proj.bias")
    if "patchembed3d.proj.weight" in out: out["patchembed3d.embedder.proj.weight"]=out.pop("patchembed3d.proj.weight").reshape(msd["patchembed3d.embedder.proj.weight"].shape)
    if "patchembed3d.proj.bias" in out: out["patchembed3d.embedder.proj.bias"]=out.pop("patchembed3d.proj.bias")
    for d in ["patchrecovery2d","patchrecovery3d"]:
        for suf in ["weight","bias"]:
            src=f"{d}.conv.{suf}"
            if src in out:
                v=out.pop(src)
                for tgt in [f"{d}.recovery.proj.{suf}",f"{d}.Reconvery.proj.{suf}"]:
                    if tgt in msd: out[tgt]=v.reshape(msd[tgt].shape) if msd[tgt].shape!=v.shape else v.clone()
    for k in list(out):
        if ".Sampler." in k:
            v=out.pop(k)
            for tgt in [k.replace(".Sampler.",".sampler."),k]:
                if tgt in msd: out[tgt]=v.clone()
    for k in list(out):
        if k.startswith("conserv."): out.pop(k)
    return out
def evalm(model):
    C=69;rmse=np.zeros(C);num=np.zeros(C);p2=np.zeros(C);l2=np.zeros(C);n=0
    with torch.no_grad():
        for data in dp.test_dataloader():
            invar,outvar=data[0],data[1]
            x=torch.cat([invar[:,:4].to(dev,torch.float32),masks,invar[:,4:].to(dev,torch.float32)],1)
            ss,su=model(x); su=su.reshape(1,65,x.shape[2],x.shape[3])
            pp=(torch.cat([ss.float(),su.float()],1)*sd_t+mu_t)[0].cpu().numpy()
            gt=(outvar.to(dev,torch.float32)*sd_t+mu_t)[0].cpu().numpy()
            rmse+=np.sqrt(((gt-pp)**2).mean(axis=(1,2))); pa=pp-clim; la=gt-clim
            num+=(pa*la).sum(axis=(1,2)); p2+=(pa**2).sum(axis=(1,2)); l2+=(la**2).sum(axis=(1,2)); n+=1
    return rmse/n, num/(np.sqrt(p2*l2)+1e-8)
TC="/tmp/teacher_real2000.npz"
z=np.load(TC); t_rmse=z["r"]; t_acc=z["a"]
def load(p):
    ck=torch.load(p,map_location=f"cuda:{dev}",weights_only=False); c=ck["cfg"]
    gm="spectral" if any("mix." in k for k in ck["model_state_dict"]) else c.get("global_mode","grid")
    st=MaxVit3DStudent(patch_size=tuple(c["patch"]),embed_dim=c["embed"],depths=tuple(c["depths"]),num_heads=tuple(c["heads"]),global_mode=gm,mlp_ratio=c.get("mlp_ratio",2.0),residual=c.get("residual",False)).to(dev).eval()
    st.load_state_dict(remap_state(ck["model_state_dict"],st.state_dict()),strict=False); return st
r_b,a_b=evalm(load(BO)); torch.cuda.empty_cache()
r_w,a_w=evalm(load(W1))
def wch(r,a,i): return min((t_rmse[i]/r[i])**2,1)+min((a[i]/t_acc[i])**2,1)
print("=== 逐通道 W 对比 (W1 vs bo3b, 单通道满分2) ===")
print(f"{'通道':30s} {'bo3b_W':>7s} {'W1_W':>7s} {'ΔW':>7s} {'bo3b_RMSE':>9s} {'W1_RMSE':>9s}")
tot_b=tot_w=0
for i in sel:
    wb=wch(r_b,a_b,i); ww=wch(r_w,a_w,i); tot_b+=wb; tot_w+=ww
    print(f"{ds.channels[i]:30s} {wb:7.4f} {ww:7.4f} {ww-wb:+7.4f} {r_b[i]:9.3f} {r_w[i]:9.3f}")
print(f"{'W15合计*20/15':30s} {tot_b/15*20:7.4f} {tot_w/15*20:7.4f} {(tot_w-tot_b)/15*20:+7.4f}")
