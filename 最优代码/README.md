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
**本提交的推理路径不依赖上述量化/剪枝库**（fp16 权重），上述库仅作为可选的进一步压缩工具列出。

学生模型自定义模块（maxvit3d_student.py）为本队原创实现，其中 stem/head 复用 OneScience 官方的
PanguEmbedding / PanguRecovery，以保证与教师输入/输出张量布局严格一致、可蒸馏、可用官方 result.py 评测。

### 无预编译二进制、HIP 源码现场编译

**提交包不含任何 `.so` / `.pyc` / `.o` 等预编译文件。**自研 HIP 算子以
`src/*.hip` 源码随包提交，由 `maxvit3d_cpp.py` 在正式计时前调用赛方镜像内的 `hipcc`
一次性编译、加载并预热；编译产物只存在于评测机临时目录，不进入提交包。

显存与速度优化使用**自研 HIP 源码 + 官方环境已有能力**达成：

- **hipBLASLt 后端**（`TORCH_BLAS_PREFER_HIPBLASLT=1`，须在 import torch 前设置）：
  PyTorch 在 DCU 上有 rocBLAS(Tensile) 与 hipBLASLt 两个 GEMM 后端，实测 hipBLASLt 的库常驻显存
  比 rocBLAS 少约 155MB（整卡峰值 1107.7MB → 953.1MB），且首次 matmul 快约 4 倍。
- **mem_opt 空间分块前向**：stem/recovery 沿纬度非重叠分块（与整块计算 bit-exact），
  stem 后立即释放输入存储，encoder/decoder 逐 block 释放中间张量。
- **分配器配置**：`expandable_segments:True` + 每样本 `empty_cache()` 抑制碎片。
- **注意力后端**：DCU 上 PyTorch SDPA 的 flash/mem_efficient backend 实测不可用
  （`No available kernel`），故走显式实现（实测更省显存）。

详见 `优化说明文档.md` 第六节。

## 二、学术诚信声明

- 学生模型**完全基于官方 Pangu 教师模型（model_bak.pth）与官方 ERA5 数据集**训练：
  第一阶段为纯教师蒸馏；第二阶段仅在 2m 温度与 13 个气压层温度上使用
  `0.5 × 教师损失 + 0.5 × 官方真值损失`，其余 55 通道始终为纯教师监督。
- **未使用任何外部气象模型、外部预训练权重、官方未授权数据或隐藏测试集**。
- **学生模型直接一次性预测完整输出场**（与输入通道数一致，覆盖全部评价与非评价通道），
  **不存在"输出 = 输入 + 增量"的残差预测路径**；训练时监督对象为完整预测场，**不做残差监督**
  （不学 delta、不用"残差 + 真值"）。网络内部 Transformer block 的层间残差连接（`x + F(x)`）属于常规结构
  设计，不涉及从模型输入到模型输出的恒等映射。
- **无任何形式的投机解码、稀疏/近似注意力、token pruning、early-exit、层跳过或通道跳过**；
  不对非评价气象指标做硬编码；不使用试错公式替代模型推理。
- 训练数据仅为赛题官方提供的 ERA5。Mitai 4090 与 abc66 5090 是本队实验室服务器，
  只存放官方同源镜像并承担计算。候选池准确为 829 个 H5（1980-1997）；最终第一阶段
  实际使用 300/13 个训练/验证样本，第二阶段使用 270/30 个连续 6h pair。归一化统计量、
  静态场（land_mask/soil_type/topography）均取自官方数据目录。
- 推理与评测**未调用任何在线服务、闭源二进制工具、外部大模型接口或不可审计的自动化工具**。
  提交包**不含任何预编译二进制文件**；HIP 源码与编译命令均随包提交并在评测机现场编译。
- 未修改赛题标注「不可更改」的计时代码与参数：`inference.py` 计时区严格沿用官方写法，
  并按官方更新在 `model(invar)` 之后、`end_time` 之前新增 `torch.cuda.synchronize()`（标注 AI4S，不可改）。
- 所有第三方代码/工具/配置均在本 README 与 `优化说明文档.md` 中标注，保证官方环境下可复现。

------------------------------------------------------------------------

## 三、目录结构

```
pangu_weather/
├── README.md                    本文件（第三方声明 + 自研 HIP 核声明 + 学术诚信 + 运行说明）
├── 宁静致远_说明文档.pdf          正式提交说明文档
├── SUBMISSION_MANIFEST.md       复核文件清单与官方外部资产边界
├── 优化说明文档.md               技术报告（轻量化方法/设计思路/性能优化/实验结果/代码模块/编译说明/运行/合规声明）
├── inference.py                 推理入口（自研 HIP 路由 + PyTorch 回退）
├── build_hip.py                 HIP 源码显式编译入口
├── verify_submission.py         提交包必要文件/源码/权重自检
├── maxvit3d_student.py          学生模型 MaxViT-3D（grid 全局注意力 + QK-Norm + RMSNorm + SwiGLU + mem_opt 分块）
├── distill_cache.py             4090 第一阶段真实脚本（官方教师纯蒸馏 + Muon/AdamW + EMA）
├── distill_truth.py             5090 第二阶段真实脚本（14 温度通道加入官方 ERA5 真值监督）
├── muon.py                      第一阶段 Muon 优化器实现
├── phys_features.py             5090 脚本的可选物理特征实现（最终配置 phys=0）
├── train.py                     官方训练脚本（保留，便于对照与教师侧复现）
├── result.py                    官方评估/可视化脚本
├── src/                         自研 HIP 完整源码
├── training/                    两阶段启动脚本、复现说明与官方变量元数据
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

# 1) 提交包完整性检查
python verify_submission.py

# 2) 可选显式编译；不执行时 inference.py 也会在正式计时前自动编译
python build_hip.py

# 3) 推理（黑盒测试入口；结果输出至 ./result/output/，时长记录至 ./result/time_record.json）
python inference.py

# 4) 评估 / 可视化
python result.py

# 5) 两阶段训练。先设置 PBASE/PSTATS/PCKPT，详见 training/README.md
bash training/run_stage1_4090.sh
bash training/run_stage2_5090.sh
```

## 许可证
官方 Pangu-Weather 源码遵循 Apache 2.0；本队学生模型实现一并以 Apache 2.0 开放。
