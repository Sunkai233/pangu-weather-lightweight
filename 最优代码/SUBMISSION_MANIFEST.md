# 提交内容清单

本目录是可独立交付的复核包，包含：

- 说明文档：`宁静致远_说明文档.pdf`。
- 轻量学生权重：`data/checkpoints/student.pth`。
- 推理与评测：`inference.py`、`result.py`、`conf/config.yaml`。
- 学生结构与推理优化：`maxvit3d_student.py`、`maxvit3d_cpp.py`。
- HIP 完整源码与编译入口：`src/*.hip`、`build_hip.py`。
- 4090 第一阶段训练：`distill_cache.py`、`muon.py`、`training/run_stage1_4090.sh`。
- 5090 第二阶段训练：`distill_truth.py`、`phys_features.py`、`training/run_stage2_5090.sh`。
- 官方变量元数据副本：`training/assets/metadata.json`。
- 完整性检查：`verify_submission.py`。

未重复打包的大文件只有组委会环境原本就提供的官方内容：Pangu 全量教师权重、ERA5 训练数据、统计量和静态场。两阶段启动脚本通过 `PCKPT`、`PBASE`、`PSTATS` 明确接入这些官方资产，并在运行前检查其存在性。项目不依赖任何外部气象模型、第三方气象数据或在线预测服务。
