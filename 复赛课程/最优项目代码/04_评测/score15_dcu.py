# -*- coding: utf-8 -*-
"""★DCU CppRunner(matrix-core libgemm) score15 真测 —— W1 权重对真值判优(最接近评测口径)。
用 ERA5Datapipe(正确归一化, 不手动 norm 避 NaN) + CppRunner fast_forward(matrix-core) +
官方15通道 W 公式(vs 教师 Pangu) + bo3b 基线对照.
用法: python score15_dcu.py <student.pth> [libgemm.so]
  默认 lib = matrix-core; 默认 student = data/checkpoints/student.pth (bo3b 基线).
"""
import os, sys, subprocess, json
os.chdir("/public/home/xdzs2026_c296/submit_pw"); sys.path.insert(0,".")
STU = sys.argv[1] if len(sys.argv)>1 else "data/checkpoints/student.pth"
LIB = sys.argv[2] if len(sys.argv)>2 else "/public/home/xdzs2026_c296/hip_dev/libgemm_mmac.so"
subprocess.run(f"cp {LIB} libgemm.so", shell=True, check=True)
os.environ["FAST_STREAM"]="1"; os.environ["ENC_FP32"]="1"; os.environ["N_TILES"]="1"; os.environ["NC_IN"]="8"; os.environ["NREC"]="16"
import numpy as np, torch
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
import maxvit3d_student as M; M.set_sdpa(False)
from maxvit3d_student import MaxVit3DStudent
dev=0
cfg=YParams("conf/config.yaml","model"); cfgd=YParams("conf/config.yaml","datapipe"); ds=cfgd.dataset
# ★数据集覆盖(env): 默认 conf 的 2012 NaN, 可指向 era5_real/2000 等正确 eval 集
_DATADIR=os.environ.get("EVAL_DATA_DIR"); _TESTYR=os.environ.get("EVAL_TEST_YEAR")
if _DATADIR:
    ds.data_dir=_DATADIR; ds.stats_dir=_DATADIR+"/stats"; ds.static_dir=_DATADIR+"/static"
if _TESTYR:
    ds.test_ratio=[int(_TESTYR)]
meta=json.load(open(os.path.join(ds.data_dir,"metadata.json"))); allvars=meta["variables"]
idx=[allvars.index(v) for v in ds.channels]
mu=np.load(os.path.join(ds.stats_dir,"global_means.npy")).astype(np.float32)[:,idx]
sd=np.load(os.path.join(ds.stats_dir,"global_stds.npy")).astype(np.float32)[:,idx]
mu_t=torch.tensor(mu,device=dev); sd_t=torch.tensor(sd,device=dev); clim=mu[0]
sdir=ds.static_dir
land=np.load(os.path.join(sdir,"land_mask.npy")).astype(np.float32)
soil=np.load(os.path.join(sdir,"soil_type.npy")).astype(np.float32)
topo=np.load(os.path.join(sdir,"topography.npy")).astype(np.float32); topo=(topo-topo.mean())/(topo.std()+1e-6)
masks=torch.tensor(np.stack([land,soil,topo],0),device=dev).unsqueeze(0)
OFFICIAL15=["mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","geopotential_850","geopotential_700","geopotential_500","specific_humidity_850","specific_humidity_700","specific_humidity_500","temperature_850","temperature_700","temperature_500","u_component_of_wind_500","v_component_of_wind_500"]
sel=[ds.channels.index(v) for v in OFFICIAL15]
dp=ERA5Datapipe(params=cfgd,distributed=False)

# teacher rmse/acc 缓存(整年 pairs)
TC="/tmp/teacher_w_cache.npz"
def teacher_ra():
    if os.path.exists(TC):
        z=np.load(TC); return z["r"],z["a"],int(z["n"])
    teacher=Pangu(img_size=ds.img_size,patch_size=cfg.patch_size,embed_dim=cfg.embed_dim,num_heads=cfg.num_heads,window_size=cfg.window_size).to(dev).eval()
    ckt=torch.load("data/checkpoints/model_bak.pth",map_location=f"cuda:{dev}",weights_only=False)
    teacher.load_state_dict(ckt["model_state_dict"] if "model_state_dict" in ckt else ckt)
    C=69;rmse=np.zeros(C);num=np.zeros(C);p2=np.zeros(C);l2=np.zeros(C);n=0
    with torch.no_grad():
        for data in dp.test_dataloader():
            invar,outvar=data[0],data[1]
            x=torch.cat([invar[:,:4].to(dev,torch.float32),masks,invar[:,4:].to(dev,torch.float32)],1)
            ss,su=teacher(x); su=su.reshape(1,65,x.shape[2],x.shape[3])
            pp=(torch.cat([ss.float(),su.float()],1)*sd_t+mu_t)[0].cpu().numpy()
            gt=(outvar.to(dev,torch.float32)*sd_t+mu_t)[0].cpu().numpy()
            rmse+=np.sqrt(((gt-pp)**2).mean(axis=(1,2))); pa=pp-clim; la=gt-clim
            num+=(pa*la).sum(axis=(1,2)); p2+=(pa**2).sum(axis=(1,2)); l2+=(la**2).sum(axis=(1,2)); n+=1
    r=rmse/n; a=num/(np.sqrt(p2*l2)+1e-8); np.savez(TC,r=r,a=a,n=n)
    del teacher,ckt; torch.cuda.empty_cache()
    return r,a,n
t_rmse,t_acc,npairs=teacher_ra()

# student via CppRunner matrix-core
ck=torch.load(STU,map_location=f"cuda:{dev}",weights_only=False); c=ck["cfg"]
gm="spectral" if any("mix." in k for k in ck["model_state_dict"]) else c.get("global_mode","grid")
m=MaxVit3DStudent(patch_size=tuple(c["patch"]),embed_dim=c["embed"],depths=tuple(c["depths"]),num_heads=tuple(c["heads"]),global_mode=gm,mlp_ratio=c.get("mlp_ratio",2.0),residual=c.get("residual",False)).to(dev).eval().half()
m.load_state_dict(ck["model_state_dict"],strict=False); m.mem_tile=6
import maxvit3d_cpp as CP
runner=CP.CppRunner(m); runner.init(); bs,bu=CP.fast_make_buffers(int(os.environ["NREC"]))
sm_h=masks.half()
C=69;rmse=np.zeros(C);num=np.zeros(C);p2=np.zeros(C);l2=np.zeros(C);n=0;fin=True
with torch.no_grad():
    for data in dp.test_dataloader():
        invar,outvar=data[0],data[1]
        full=torch.cat([invar[:,:4].half(),sm_h.cpu(),invar[:,4:].half()],1)  # CPU [1,72,H,W]
        pk=CP.prepack_input(full); ms,mu_=runner.fast_forward(pk,bs,bu); torch.cuda.synchronize()
        ss=CP.reassemble_fast(ms,bs).squeeze(2); su=CP.reassemble_fast(mu_,bu).reshape(1,65,full.shape[2],full.shape[3])
        pred=torch.cat([ss.float(),su.float()],1)
        if not torch.isfinite(pred).all(): fin=False
        pp=(pred.to(dev)*sd_t+mu_t)[0].cpu().numpy()
        gt=(outvar.to(dev,torch.float32)*sd_t+mu_t)[0].cpu().numpy()
        rmse+=np.sqrt(((gt-pp)**2).mean(axis=(1,2))); pa=pp-clim; la=gt-clim
        num+=(pa*la).sum(axis=(1,2)); p2+=(pa**2).sum(axis=(1,2)); l2+=(la**2).sum(axis=(1,2)); n+=1
        del pk,ms,mu_; torch.cuda.empty_cache()
sr=rmse/n; sa=num/(np.sqrt(p2*l2)+1e-8)
def W(ix): return float(np.mean(np.minimum((t_rmse[ix]/sr[ix])**2,1)+np.minimum((sa[ix]/t_acc[ix])**2,1))*20)
print(f"=== DCU CppRunner(matrix-core) score15 真测 ===")
print(f"student={STU.split('/')[-1]} pairs={n} finite={fin}")
print(f"W15={W(sel):.4f}  W69={W(list(range(69))):.4f}  RMSE15={sr[sel].mean():.4f}  ACC15={sa[sel].mean():.4f}")
print(f"bo3b基线 W15=37.18 (score15) → {'★超基线!' if W(sel)>37.18 else '低于基线' if W(sel)<37.18 else '持平'}")
