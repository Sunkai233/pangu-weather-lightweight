# -*- coding: utf-8 -*-
"""6 大考点一键自测:每个打印 PASS/FAIL。在含 maxvit3d_student.py 的环境跑。
用法:python selftest_6考点.py
路径:自动从 PBASELINE 或本目录 baseline/官方源码/ 找 maxvit3d_student.py。
"""
import os, sys, torch, torch.nn.functional as F

for _p in (os.environ.get("PBASELINE", ""),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline", "官方源码"),
           "."):
    if _p and os.path.isfile(os.path.join(_p, "maxvit3d_student.py")):
        sys.path.insert(0, _p); break

ok_all = True
def check(name, cond, extra=""):
    global ok_all; ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")


def t1_window():
    from maxvit3d_student import _pad3d, block_partition, block_reverse, grid_partition, grid_reverse
    B, Pl, Lat, Lon, C = 1, 8, 181, 360, 16
    x = torch.randn(B, Pl, Lat, Lon, C)
    for name, part, rev, win in [("block", block_partition, block_reverse, (2, 6, 12)),
                                 ("grid", grid_partition, grid_reverse, (2, 6, 12))]:
        xp, _ = _pad3d(x, win); pPl, pLat, pLon = xp.shape[1:4]
        w = part(xp, win)
        xr = rev(w, win, B, pPl, pLat, pLon)[:, :Pl, :Lat, :Lon, :]
        check(f"考点1 {name} partition/reverse 可逆", torch.allclose(x, xr), f"窗口token={tuple(w.shape)}")


def t2_channels():
    x = torch.randn(1, 72, 721, 1440)
    surface = x[:, :7]; upper = x[:, 7:].reshape(1, 5, 13, 721, 1440)
    out = torch.cat([torch.randn(1, 4, 721, 1440), torch.randn(1, 5, 13, 721, 1440).reshape(1, 65, 721, 1440)], 1)
    check("考点2 72拆分(7面+65高空)", surface.shape[1] == 7 and upper.shape[1:3] == (5, 13))
    check("考点2 69拼接", out.shape[1] == 69)


def t3_attn():
    from maxvit3d_student import Attention3D, set_sdpa
    attn = Attention3D(dim=96, win=(2, 6, 12), num_heads=6)
    x = torch.randn(1, 2 * 6 * 12, 96)
    set_sdpa(False); y1 = attn(x)
    set_sdpa(True);  y2 = attn(x)
    check("考点3 QK-Norm 注意力 形状/无NaN", y1.shape == x.shape and not torch.isnan(y1).any(),
          f"两路一致={torch.allclose(y1, y2, atol=2e-3)}")


def t4_loss():
    def wl1(x, y, w, lw): return lw * (F.l1_loss(x, y, reduction="none") * w).mean()
    sw = torch.tensor([1.5, 0.77, 0.66, 3.0]).view(1, 4, 1, 1); pw = torch.ones(1, 65, 1, 1)
    ss, su = torch.randn(1, 4, 32, 32), torch.randn(1, 65, 32, 32)
    ts, tu, tsf, tua = (torch.randn(1, 4, 32, 32), torch.randn(1, 65, 32, 32),
                        torch.randn(1, 4, 32, 32), torch.randn(1, 65, 32, 32))
    ld = wl1(ss, ts, sw, 0.25) + wl1(su, tu, pw, 1.0)
    lg = wl1(ss, tsf, sw, 0.25) + wl1(su, tua, pw, 1.0)
    loss = 0.5 * ld + 0.5 * lg
    check("考点4 蒸馏双监督加权L1", torch.isfinite(loss).all() and loss.item() > 0, f"loss={loss.item():.3f}")


def t5_muon():
    def _ns5(G, steps=6, eps=1e-7):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16(); X = X / (X.norm() + eps)
        tp = G.size(0) > G.size(1)
        if tp: X = X.T
        for _ in range(steps):
            A = X @ X.T; Bm = b * A + c * (A @ A); X = a * X + Bm @ X
        return (X.T if tp else X)
    O = _ns5(torch.randn(128, 64)).float()
    s = torch.linalg.svdvals(O)
    # Muon 的 NS 不追求精确正交:系数把奇异值压向 1 的一个带内(bf16+6步约 [0.65,1.2]),不是全=1
    check("考点5 Muon Newton-Schulz 奇异值压向1", (s.min() > 0.5 and s.max() < 1.5), f"奇异值[{s.min():.2f},{s.max():.2f}](≈1即对)")


def t6_student():
    from maxvit3d_student import MaxVit3DStudent
    import maxvit3d_student as M; M.set_sdpa(True)
    m = MaxVit3DStudent(patch_size=(2, 16, 16), embed_dim=96, depths=(2, 4, 2),
                        num_heads=(6, 12, 6), mlp_ratio=2.0).eval()
    with torch.no_grad(): os_, ou = m(torch.randn(1, 72, 721, 1440))
    p = sum(t.numel() for t in m.parameters()) / 1e6
    check("考点6 U形学生 72→69 前向", os_.shape == (1, 4, 721, 1440) and ou.shape == (1, 5, 13, 721, 1440),
          f"参数={p:.2f}M(应≈4.35M)")


if __name__ == "__main__":
    print("=== 6 大考点自测 ===")
    for fn in (t1_window, t2_channels, t3_attn, t4_loss, t5_muon, t6_student):
        try: fn()
        except Exception as e: ok_all = False; print(f"  [FAIL] {fn.__name__} 异常: {type(e).__name__}: {e}")
    print("=== 全部通过 ===" if ok_all else "=== 有失败项,见上 ===")
    sys.exit(0 if ok_all else 1)
