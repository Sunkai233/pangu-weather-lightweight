# -*- coding: utf-8 -*-
"""Muon 优化器(Keller Jordan)自包含实现,纯torch,不依赖pip install。
用法:2D权重(attn/mlp的weight)走Muon,embedding/norm/bias/conv走AdamW。
  muon_params  = [p for n,p in model.named_parameters() if p.ndim==2 and 'embed' not in n and 'recovery' not in n and 'patchembed' not in n]
  adamw_params = [其余]
  opt_m = Muon(muon_params, lr=0.02, momentum=0.95)
  opt_a = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9,0.95), weight_decay=0.05)
  # step: opt_m.step(); opt_a.step();  两个都zero_grad
参考:https://github.com/KellerJordan/Muon ; arXiv 2509.15816 / 2505.02222(小模型30-40%提速)
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz 5次迭代,把矩阵G正交化(近似 G@(G^T G)^-0.5)。bf16省算力。"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T; transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """只放 2D 权重矩阵。lr 通常比 AdamW 大一个量级(0.01~0.02)。"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]; mom = group["momentum"]; nest = group["nesterov"]; ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)          # 兜底:高维reshape成2D
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                buf = st["mom"]
                buf.lerp_(g, 1 - mom)                   # 动量EMA
                upd = g.lerp_(buf, mom) if nest else buf  # nesterov
                upd = zeropower_via_newtonschulz5(upd, steps=ns)
                upd = upd * (max(1.0, g.size(0) / g.size(1)) ** 0.5)  # 形状感知缩放
                p.add_(upd.view_as(p).to(p.dtype), alpha=-lr)
        return loss
