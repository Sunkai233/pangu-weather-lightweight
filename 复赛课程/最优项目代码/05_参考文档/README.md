# Pangu-Weather 轻量化与推理优化（先导杯 2026 AI4S 赛题二 · 国产 DCU）

本提交在国产海光 DCU（hygon）上，对官方 Pangu-Weather 气象大模型做**知识蒸馏轻量化 + 推理显存/速度优化**。
学生模型 MaxViT-3D 仅 5.19M 参数（教师 64M），推理时长 65.7ms（教师 979.9ms，16.8× 加速），
跨样本峰值显存 548MB（教师 12.5GB），权重以 fp16 半精度存储仅 10.46MB（教师 1.87GB），
与教师推理精度高度对齐（官方 15 指标口径 W=35.63 / 满分 40）。

------------------------------------------------------------------------

## 一、第三方库与运行环境声明

本方案在赛题官方环境内运行，**未引入任何官方环境之外的第三方依赖**。所用库均为官方镜像/赛题约定环境自带：

- OneScience 0.3.0（官方科学计算框架，提供 Pangu 模型、ERA5Datapipe、YParams、PanguEmbedding/Recovery 等）
- Python 3.11 / PyTorch 2.5.1（DCU 版）
- numpy / h5py / tqdm（基础数据与 IO，官方环境自带）

量化与图优化能力（如需进一步压缩，文档中给出做法）所对应的库同属赛题约定环境，均为官方提供、可在官方环境复现：
onnx 1.21.0 / onnxsim 0.6.3 / onnxconverter-common 1.16.0 / tensorly 0.9.0 / torch-pruning 1.6.1。
**本提交的推理路径本身不依赖上述量化/剪枝库**（fp16 权重 + autocast，纯 PyTorch 即可运行），上述库仅作为可选的进一步压缩工具列出。

学生模型自定义模块（maxvit3d_student.py）为本队原创实现，其中 stem/head 复用 OneScience 官方的
PanguEmbedding / PanguRecovery，以保证与教师输入/输出张量布局严格一致、可蒸馏、可用官方 result.py 评测。

## 二、学术诚信声明

- 学生模型**完全基于官方 Pangu 教师模型（model_bak.pth）与官方 ERA5 数据集**通过知识蒸馏 + 真值双监督训练得到。
- **未使用任何外部气象模型、外部预训练权重、官方未授权数据或隐藏测试集**。
- 训练数据仅为赛题官方提供的 ERA5（`onedatasets/ERA5_test`），归一化统计量、静态场（land_mask/soil_type/topography）均取自官方数据目录。
- 推理与评测**未调用任何在线服务、闭源二进制工具、外部大模型接口或不可审计的自动化工具**。
- 未修改赛题标注「不可更改」的计时代码与参数：`inference.py` 计时区严格沿用官方写法，
  并按官方更新在 `model(invar)` 之后、`end_time` 之前新增 `torch.cuda.synchronize()`（标注 AI4S，不可改）。
- 所有第三方代码/工具/配置均在本 README 与 `优化说明文档.md` 中标注，保证官方环境下可复现。

------------------------------------------------------------------------

## 三、目录结构

```
pangu_weather/
├── README.md                    本文件（第三方声明 + 学术诚信 + 运行说明）
├── 优化说明文档.md               技术报告（轻量化方法/设计思路/性能优化/实验结果/代码模块/编译/运行）
├── inference.py                 推理入口（含显存优化：SDPA off + 空间分块 tiling + fp16 输入 + autocast）
├── maxvit3d_student.py          学生模型 MaxViT-3D（grid 全局注意力 + QK-Norm + RMSNorm + SwiGLU + mem_opt 分块）
├── distill_scnet.py             蒸馏训练主脚本（AdamW + warmup 余弦 + EMA + bf16 + 梯度检查点 + 双监督 + 最优 ckpt）
├── distill_train.py             蒸馏训练精简脚本（深监督 DTP 双损失，便于快速复现）
├── train.py                     官方训练脚本（保留，便于对照与教师侧复现）
├── result.py                    官方评估/可视化脚本
├── fake_data.py                 伪造 ERA5，用于无真实数据时快速跑通流程
├── conf/
│   └── config.yaml             配置（datapipe 通道/归一化路径/img_size；test_ratio 已置占位年份）
└── data/
    └── checkpoints/
        └── student.pth         学生轻量权重（fp16，10.46MB；内含 model_state_dict + 学生结构 cfg）
```

> 教师权重 model_bak.pth、ERA5 数据集、统计量、静态场均由赛题官方环境提供（位于 `../onedatasets` 与官方 checkpoints），
> **不打入本提交包**，故解压目录精简（仅推理必需的代码 + 配置 + 学生权重）。

## 四、运行使用说明

```bash
source ../earth_env.sh          # 加载官方环境

# 1) 推理（黑盒测试入口；结果输出至 ./result/output/，时长记录至 ./result/time_record.json）
python inference.py

# 2) 评估 / 可视化
python result.py

# 3)（可选）复现蒸馏训练
python distill_scnet.py --embed 96 --depths 2,4,2 --heads 6,12,6 --patch 2,16,16 \
                        --global-mode grid --mlp-ratio 2.0 --epochs 120 --alpha 0.5
```

无真实数据时可先用伪造数据跑通流程：
```bash
python fake_data.py             # 生成虚拟 ERA5（仅验证流程，可忽略）
```

## 许可证
官方 Pangu-Weather 源码遵循 Apache 2.0；本队学生模型实现一并以 Apache 2.0 开放。
