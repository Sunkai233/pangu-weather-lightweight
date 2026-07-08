# -*- coding: utf-8 -*-
"""★W 权重判优 harness: bo3b/W1 在 era5_real/2000 纯 fp32 真测 score15(vs 教师 Pangu).
fp32 forward(bo3b fp16 会 NaN, fp32 finite). 这是 W 判优最准口径(38.44 参考是 fp32).
用法: python score15_fp32.py <student.pth>  (默认 bo3b)
"""
import os, sys, json
os.chdir("/public/home/xdzs2026_c296/submit_pw"); sys.path.insert(0,".")
STU = sys.argv[1] if len(sys.argv)>1 else "data/checkpoints/student.pth"
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
    return rmse/n, num/(np.sqrt(p2*l2)+1e-8), n

# teacher 缓存
TC="/tmp/teacher_real2000.npz"
if os.path.exists(TC):
    z=np.load(TC); t_rmse=z["r"]; t_acc=z["a"]; npairs=int(z["n"])
else:
    teacher=Pangu(img_size=ds.img_size,patch_size=cfg.patch_size,embed_dim=cfg.embed_dim,num_heads=cfg.num_heads,window_size=cfg.window_size).to(dev).eval()
    ckt=torch.load("data/checkpoints/model_bak.pth",map_location=f"cuda:{dev}",weights_only=False)
    teacher.load_state_dict(ckt["model_state_dict"] if "model_state_dict" in ckt else ckt)
    t_rmse,t_acc,npairs=evalm(teacher); np.savez(TC,r=t_rmse,a=t_acc,n=npairs); del teacher,ckt; torch.cuda.empty_cache()

ck=torch.load(STU,map_location=f"cuda:{dev}",weights_only=False); c=ck["cfg"]
gm="spectral" if any("mix." in k for k in ck["model_state_dict"]) else c.get("global_mode","grid")
st=MaxVit3DStudent(patch_size=tuple(c["patch"]),embed_dim=c["embed"],depths=tuple(c["depths"]),num_heads=tuple(c["heads"]),global_mode=gm,mlp_ratio=c.get("mlp_ratio",2.0),residual=c.get("residual",False)).to(dev).eval()  # fp32
def remap_state(sdk, msd):
    """distill_truth.py 命名 → MaxVit3DStudent 命名(已匹配则原样).
    embed proj→embedder.proj(2d补kz维); recovery conv→recovery.proj+Reconvery.proj(双填);
    Sampler→sampler+Sampler(双填); 丢 conserv.*。"""
    if "patchembed2d.embedder.proj.weight" in sdk:
        return sdk
    out=dict(sdk)
    if "patchembed2d.proj.weight" in out:
        w=out.pop("patchembed2d.proj.weight"); out["patchembed2d.embedder.proj.weight"]=w.reshape(msd["patchembed2d.embedder.proj.weight"].shape)
    if "patchembed2d.proj.bias" in out: out["patchembed2d.embedder.proj.bias"]=out.pop("patchembed2d.proj.bias")
    if "patchembed3d.proj.weight" in out:
        w=out.pop("patchembed3d.proj.weight"); out["patchembed3d.embedder.proj.weight"]=w.reshape(msd["patchembed3d.embedder.proj.weight"].shape)
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
            for tgt in [k.replace(".Sampler.",".sampler."), k]:
                if tgt in msd: out[tgt]=v.clone()
    for k in list(out):
        if k.startswith("conserv."): out.pop(k)
    return out
sdk2=remap_state(ck["model_state_dict"], st.state_dict())
r=st.load_state_dict(sdk2,strict=False)
if len(r.missing_keys)>0:
    print(f"[警告] load 后仍缺 {len(r.missing_keys)} key(随机init): {r.missing_keys[:5]}")
sr,sa,n=evalm(st)
def W(ix): return float(np.mean(np.minimum((t_rmse[ix]/sr[ix])**2,1)+np.minimum((sa[ix]/t_acc[ix])**2,1))*20)
print(f"=== score15 fp32 @ era5_real/2000 (pairs={n}) ===")
print(f"student={STU.split('/')[-1]}")
print(f"W15={W(sel):.4f}  W69={W(list(range(69))):.4f}  RMSE15={sr[sel].mean():.4f}  ACC15={sa[sel].mean():.4f}")
print(f"bo3b基线≈38.44 → {'★超基线!' if W(sel)>38.44 else '低于' if W(sel)<38.44-0.005 else '持平'}")
