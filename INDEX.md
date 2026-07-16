# 气象大模型轻量化 · 最终成果索引

## 版本与成绩

- 赛题：先导杯 2026 AI4S 赛题二「气象大模型轻量化」。
- 教师：组委会官方 Pangu-Weather，64.22M 参数，训练中冻结。
- 学生：MaxViT-3D，约 5.19M 参数，FP16 权重约 10.46MB。
- 历史官方判卷：96.1213。
- 当前同权重推理优化提交实测：96.5686（U 38.9323 / V 19.1978 / W 38.4387）。
- 本索引对应 2026-07-16 GitHub 更新。

## 最终训练血缘

1. `最优代码/training/run_stage1_4090.sh`
   - 实验室 Mitai 4090。
   - 1980-1996 年官方 ERA5 候选数据训练，1997 年验证。
   - 69 通道纯官方教师蒸馏。
   - `embed=96`、`depths=(2,4,2)`、`heads=(6,12,6)`、`patch=(2,16,16)`、`global_mode=grid`。
   - Muon `lr=0.02, momentum=0.99` + AdamW `lr=6e-4`，BF16、EMA、经度 roll。

2. `最优代码/training/run_stage2_5090.sh`
   - 实验室 abc66 5090。
   - 从第一阶段权重 `strict=True` 续训。
   - 2m 温度和 13 个气压层 temperature 使用 `0.5×教师 + 0.5×官方 ERA5 真值`；其余 55 通道仍为纯教师监督。
   - `lr=6e-5`、warmup 2、BF16；最终采用 epoch 21 冻结快照。

最终权重不使用输出残差预测；DTP、物理特征头、频谱损失、全 15 指标真值、额外风湿真值、区域重加权和冻结 backbone 均关闭。

## 主要目录

### `最优代码/`

当前正式提交目录的解压版：

- 推理与评测：`inference.py`、`result.py`、`conf/config.yaml`。
- 模型：`maxvit3d_student.py`、`maxvit3d_cpp.py`。
- 两阶段训练：`distill_cache.py`、`distill_truth.py`、`muon.py`、`phys_features.py`、`training/*.sh`。
- HIP 源码与编译：`src/attn_lib.hip`、`src/gemm_lib_mmac.hip`、`build_hip.py`。
- 权重：`data/checkpoints/student.pth`。
- 复核：`verify_submission.py`、`SUBMISSION_MANIFEST.md`。
- 文档：`宁静致远_说明文档.pdf`、`优化说明文档.md`、`README.md`。

提交内容不含预编译 `.so/.o/.dll`；HIP 在赛方预热阶段由源码现场编译。

### `最终提交包/`

- `pangu_weather.zip`：可直接提交/归档的精确压缩包。
- `宁静致远_说明文档.pdf`：正式队伍说明文档。

### `升级优化10讲/`

- `00_课程总览.html`：课程入口。
- L1-L10：从 U/V/W 评分、Pangu 教师、学生骨架、模型/优化器对比，到蒸馏、真值监督、失败分支、显存和 DCU 部署。
- 第 4 讲已更新为最终网络、Muon + AdamW 和两阶段训练的系统对照。

### `复赛课程/` 与 `官方参考/`

保留教学、历史实验和官方教师结构资料；其中探索脚本不代表最终权重配置，最终配置以 `最优代码/training/*.sh` 和权重内嵌 `cfg` 为准。

## 复现边界

仓库内提供全部自研训练、推理、优化和编译源码，以及学生权重。以下大文件由组委会环境提供，因此不重复提交：

- Pangu 全量教师权重 `model_bak.pth`。
- 官方 ERA5 数据集。
- 官方归一化统计量与 `land_mask / soil_type / topography` 静态场。

两阶段启动脚本通过 `PBASE`、`PSTATS`、`PCKPT`、`PMETA` 显式接入并检查这些官方资产。
