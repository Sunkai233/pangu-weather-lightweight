# -*- coding: utf-8 -*-
"""DCU 训练步微基准:找最快配置。教师fwd + 学生step(fwd+bwd) under {materialized/SDPA}x{ckpt on/off}。"""
import torch, time
import maxvit3d_student as M
from maxvit3d_student import MaxVit3DStudent
from onescience.models.pangu import Pangu
dev = 0
torch.backends.cuda.matmul.allow_tf32 = True
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
def rand(): return torch.randn(1, 72, 721, 1440, device=dev)
def sync(): torch.cuda.synchronize()
CKPT = "/public/home/xdzs2026_c296/xiandao2026-AI4S/pangu_weather/data/checkpoints/model_bak.pth"

teacher = Pangu(img_size=(721, 1440)).to(dev).eval()
ck = torch.load(CKPT, map_location=f"cuda:{dev}", weights_only=False)
teacher.load_state_dict(ck["model_state_dict"])
for p in teacher.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    for _ in range(2):
        with amp(): teacher(rand())
    sync(); t = time.time()
    for _ in range(5):
        with amp(): teacher(rand())
    sync(); tf = (time.time() - t) / 5 * 1000
print(f"教师 fwd(no_grad,bf16): {tf:.0f} ms/it", flush=True)
print(f"显存 after teacher: {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)

for sdpa in [False, True]:
    M.set_sdpa(sdpa)
    for ckpt in [False, True]:
        try:
            torch.cuda.reset_peak_memory_stats()
            st = MaxVit3DStudent(patch_size=(2, 4, 4), embed_dim=64, depths=(1, 3, 1),
                                 num_heads=(4, 8, 4), use_checkpoint=ckpt).to(dev)
            opt = torch.optim.AdamW(st.parameters(), lr=1e-4)
            def step():
                x = rand()
                with amp():
                    ss, su = st(x); loss = ss.float().mean() + su.float().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for _ in range(2): step()
            sync(); t = time.time()
            for _ in range(5): step()
            sync(); ms = (time.time() - t) / 5 * 1000
            mem = torch.cuda.max_memory_allocated() / 1e9
            full = tf + ms
            print(f"学生step sdpa={int(sdpa)} ckpt={int(ckpt)}: {ms:.0f} ms  | 全步(教师+学生)≈{full:.0f} ms  | 峰值{mem:.1f}G", flush=True)
            del st, opt; torch.cuda.empty_cache()
        except Exception as e:
            print(f"sdpa={sdpa} ckpt={ckpt} FAIL: {repr(e)[:120]}", flush=True)
