# 盘古气象大模型轻量化（先导杯赛题二 · 国产海光 DCU）

本仓库保存 Pangu-Weather 轻量化项目的最终代码、训练链路、学生权重、HIP 源码、课程讲义与正式提交快照。

当前版本更新时间：**2026-07-16**。历史官方判卷版本为 **96.1213**；在保持同一学生模型与权重的基础上继续做数值等价推理优化后，当前提交实测为 **96.5686**（U 38.9323、V 19.1978、W 38.4387）。

## 最终方案

- 教师：组委会提供的 Pangu-Weather，64.22M 参数，训练学生时全程冻结。
- 学生：MaxViT-3D，约 5.19M 参数，FP16 权重约 10.46MB。
- 结构：`patch=(2,16,16)`、`embed=96`、`depths=(2,4,2)`、`heads=(6,12,6)`、block/grid 交替注意力、QK-Norm、RMSNorm、SwiGLU、LayerScale。
- 第一阶段：实验室 4090 上进行 69 通道纯教师蒸馏，Muon + AdamW、BF16、EMA。
- 第二阶段：实验室 5090 上严格续训，仅对 2m 温度和 13 个气压层温度加入官方 ERA5 真值监督，其余 55 通道保持教师监督。
- 输出：直接预测完整 t+6h 绝对气象场，`residual=False`。
- 最终关闭：DTP、物理特征头、频谱损失、全 15 指标真值监督、额外风湿真值、区域重加权、冻结 backbone。
- 部署：随包提交完整 HIP 源码，评测机预热阶段现场编译；仓库不再保存预编译 `.so`。

## 目录

- `最优代码/`：与当前正式提交包一致的可复核目录，含训练、推理、编译、评测、权重和说明文档。
- `最优权重/student_fp16_10MB.pth`：为兼容旧路径保留的最终学生权重副本。
- `最终提交包/`：2026-07-16 的精确提交 ZIP 与队伍说明 PDF。
- `升级优化10讲/`：逐模块课程；入口为 `00_课程总览.html`，第 4 讲系统对比 Pangu 与学生网络、Muon/AdamW 及训练策略。
- `复赛课程/`：教学材料与历史实验代码。
- `官方参考/`：Pangu 教师结构和 ERA5 处理参考。
- `INDEX.md`：最终配置、训练血缘和复现路径索引。

## 快速复核

```bash
cd 最优代码
python verify_submission.py
python build_hip.py     # 需要赛方 DTK/ROCm/DCU 环境
python inference.py     # 需要赛方 OneScience、ERA5 测试资产和 DCU
python result.py
```

训练入口：

```bash
cd 最优代码
PBASE=/path/to/official/era5 \
PSTATS=/path/to/official/ERA5_test \
PCKPT=/path/to/official/model_bak.pth \
bash training/run_stage1_4090.sh

PBASE=/path/to/official/continuous_era5 \
PSTATS=/path/to/official/ERA5_test \
PCKPT=/path/to/official/model_bak.pth \
bash training/run_stage2_5090.sh
```

教师权重、完整 ERA5、归一化统计量和静态场由组委会环境提供，不重复打入仓库。训练数据均为组委会官方 ERA5 同源副本；未使用外部气象模型、外部预训练权重、隐藏测试集或未经授权数据。

详细方法、逐项合规说明和完整文件清单见 `最终提交包/宁静致远_说明文档.pdf` 与 `最优代码/SUBMISSION_MANIFEST.md`。
