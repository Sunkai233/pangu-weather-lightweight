# 两阶段训练复现

本目录提供最终学生权重对应的两阶段启动入口。训练代码本体位于项目根目录：

- `distill_cache.py`：实验室 Mitai 4090 第一阶段，使用官方冻结 Pangu 教师做纯蒸馏。
- `distill_truth.py`：实验室 abc66 5090 第二阶段，从第一阶段权重续训；仅 14 个温度通道加入官方 ERA5 真值监督。
- `muon.py`：第一阶段使用的 Muon 优化器。
- `maxvit3d_student.py`：训练与推理共同使用的学生结构。

训练前必须提供下列组委会官方资产：

1. `PBASE`：官方 ERA5 根目录，其下为 `data/<year>/*.h5`。
2. `PSTATS`：官方统计量和静态场根目录，其下有 `stats/` 与 `static/`。
3. `PCKPT`：组委会提供的 Pangu 全量教师权重 `model_bak.pth`。
4. `PMETA`：变量顺序元数据；默认使用随包提供的 `training/assets/metadata.json`。

示例：

```bash
source ../earth_env.sh

PBASE=/path/to/official/era5 \
PSTATS=/path/to/official/ERA5_test \
PCKPT=/path/to/official/model_bak.pth \
bash training/run_stage1_4090.sh

PBASE=/path/to/official/continuous_era5 \
PSTATS=/path/to/official/ERA5_test \
PCKPT=/path/to/official/model_bak.pth \
bash training/run_stage2_5090.sh
```

两份启动脚本均在真正启动前机械检查数据目录、统计量、元数据、教师权重和续训权重，缺失时立即退出，不会静默改用其他数据或权重。
