# DCU 端到端训练复刻(海光 gfx936 / SCNet)

这是在**国产海光 DCU** 上对提交学生模型(MaxViT-3D,4.349M,教师 64.22M 的 6.8%)做的**端到端训练复刻**,用于证明整条轻量化训练管线可在国产硬件上原生跑通(复赛强调"全量模型轻量化 + 国产硬件推理部署效率")。

## 结果

| 项 | DCU 复刻(本目录) | 5090 原版(实际提交) |
|---|---|---|
| 硬件 | 海光 DCU gfx936 | RTX 5090 |
| 训练样本 | 900 train / 24 val | 250 train / 200 val(1980) |
| 轮数 | 60 | 80 |
| 优化器 | AdamW | **Muon**(83 个 2D 权重)+ AdamW |
| 频域蒸馏 | 关(spec 0) | 关 |
| best val_gtL1 | **0.0966** | — |
| best val_match | **92.1%** | 92.97% |

复刻权重 `student_dcu_repro.pth`(SHA256 `0a7cdd7f…b1f2`)与提交权重 **cfg 九项完全一致、strict=True 可加载、250 个张量形状全对齐**。

> 差距 0.87 个百分点主要来自 5090 版用了 **Muon + 多 20 轮**:DCU 复刻期间容器被平台回收两次,续训必须复用 AdamW 的 `.ckpt`(换 Muon 无法 resume),故锁定 AdamW。这反证了 Muon 是原版调优的主要功臣。架构 / 蒸馏损失(α=0.5,教师+ERA5真值各半)/ lr 6e-4 / warmup 5 / 经度增广均一致。

## 文件

- `pangu_weather/distill_dcu.py` —— DCU 蒸馏主训练(自实现 Muon、频域蒸馏、int8 缓存流式读取、断点续训)
- `pangu_weather/prep_cache.py` —— 预处理:教师前向 + ERA5 真值 → int8 逐通道量化落盘(216MB/样本),只跑一次
- `pangu_weather/make_config_train.py` —— 从官方 `config.yaml` 派生训练配置(不改官方文件)
- `pangu_weather/conf/config_train.yaml` —— 派生出的训练配置
- `run_train_fast.sh` —— 快速路径启动脚本(读 int8 缓存,反复执行即从 `.ckpt` 续训)
- `train_dcu.log` / `prep.log` —— 真实训练 / 预处理日志
- `student_dcu_repro.pth` —— 复刻权重(fp32,17MB)

## 复现步骤(DCU 容器内)

```bash
# 1) 预处理:教师输出 + 真值 → int8 缓存(约 183GB,只需一次)
python prep_cache.py --config conf/config_train.yaml \
  --cache-dir /path/cache_int8 --split train
# 2) 训练:标准 SGD 全量遍历,每 epoch 随机;断点续训
bash run_train_fast.sh
```

## 合规

学生模型自设计架构 + 随机初始化,仅从官方教师权重 `model_bak.pth` 蒸馏(官方权重只读引用,未改官方源码);无第三方权重、无残差预测、无炼丹、无预编译二进制。
