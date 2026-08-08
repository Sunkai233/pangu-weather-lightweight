# 共享数据训练实测 + 决赛三赛题解题

用**集群共享盘的真实 ERA5**、在**海光 DCU 容器内存受限**下,跑通学生蒸馏训练;并按决赛说明推断三类赛题,在 DCU 上**真跑验证**。全部结果为 2026-08-08 在 SCNet DCU 容器实测。

## 文档(打印用 HTML,白底)
- **[`训练实测文档.html`](训练实测文档.html)** —— ★真实共享数据怎么读进容器 + 内存墙(容器 cgroup 63.8G)+ 落盘缓存优化 + 250 样本训练实测(收敛曲线)。
- **[`决赛三赛题解题文档.html`](决赛三赛题解题文档.html)** —— 按"增加任务难度 / 全量模型轻量化 / 国产硬件部署效率"推断三赛题,各配 DCU 实测。
- `实测日志.txt` —— DCU 上训练 + 三题测试的原始日志(去噪)。

## 代码
- `代码/distill_diskcache_scnet.py` —— ★本次实测用:落盘缓存蒸馏(适配 63.8G 容器,几百样本)。
- `代码/distill_scnet_fast_RAM缓存原版.py` —— RAM 缓存原版(几百样本会 OOM,对照)。
- `代码/finals_tests.py` —— 决赛三赛题一键测试(`python finals_tests.py 1|2|3`)。

## 关键结果(一眼看全)
**训练**(250 真实共享样本 1980 年 + 39 验证 1981 年,学生 1.70M=教师 2.7%):
- 落盘缓存 72GB + 验证 17GB RAM,训练峰值 RAM ~20G < 63.8G 容器上限 ✓(RAM 缓存版需 147G→OOM)
- 8 epoch 干净收敛:val_gtL1 **1.20→0.35**,val_match **7%→70.7%**,每 epoch ~115s

**决赛三赛题 DCU 实测**:
- ① 多步自回归:归一化 RMSE t+6h 0.133 → t+24h **0.260**(误差累积,任务更难)
- ② 全模型 INT8:256.9MB → 64.9MB(**3.96×**),预测差 RMSE 仅 **0.00964**(近无损)
- ③ fp16 推理:949.4ms → 715.6ms(**1.33×,-25%**),显存 10.32→5.47GB(**-47%**)

## 怎么复现(SCNet DCU 容器)
1. **共享数据进容器**(宿主机 `--host`):从 `/work2/share/sugonhpcapp01/ERA5/old-data/data/{年}` 拷所需时刻到 `/public/home/.../era5_big/data/{年}`;`stats/static/metadata` 复用 `era5_real` 的。
2. **配 config**:`data_dir=era5_big`、`train_ratio=[1980]`、`val_ratio=[1981]`;软链教师 `data/checkpoints/model_bak.pth`。
3. **训练**:`python distill_diskcache_scnet.py --limit-train 250 --epochs 8 --embed 64 --patch 2,8,8`
4. **三题测试**:`python finals_tests.py 1 && python finals_tests.py 2 && python finals_tests.py 3`

> 扩到几千样本:落盘缓存下 RAM 恒定,只受磁盘配额(每样本 ~287MB);加 `--limit-train`/年份即可。
> 容器数据占用:era5_big≈131G + tcache≈72G(可删,已训完)。
