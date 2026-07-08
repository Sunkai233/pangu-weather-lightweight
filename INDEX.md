# 气象大模型轻量化(先导杯赛题二)· 最优成果索引

## 赛题与成绩
- **赛题**:先导杯 2026 AI4S 赛题二「气象大模型轻量化」(国产海光 DCU)
- **教师**:Pangu-Weather(64.22M 参数,权重 1.8G)
- **最优方案**:MaxViT-3D U 形学生 + 知识蒸馏 + 真值双监督(DTP)
- **最优成绩**:官方 15 指标 **96.1213 分**
- **学生规格**:5.19M 参数(教师 8.1%)/ 10.46MB fp16 权重 / 推理 65.7ms(14.9× 加速)/ 峰值显存 548MB

## 本文件夹内容(本地最优,复制自原目录)

### 最优代码/(推理部署 + 蒸馏,来自最终提交包 提交_pangu_weather/pangu_weather/)
- `maxvit3d_student.py` — **★最优学生模型**(MaxViT-3D:grid 等距注意力 + QK-Norm + RMSNorm + SwiGLU)
- `maxvit3d_cpp.py` — C++ 部署版学生模型
- `inference.py` — **★推理入口**(黑盒评测:SDPA off + 分块 tiling + fp16 输入 + autocast 显存优化)
- `distill_scnet.py` — **★蒸馏主脚本**(AdamW + EMA + bf16 混精 + 双监督 DTP)
- `distill_train.py` — 蒸馏精简版(快速复现)
- `result.py` — 官方评估/可视化
- `优化说明文档.md` — 完整技术报告(模型设计 + 显存优化 + 实验结果)
- `conf/config.yaml` — 配置;`libflashattn.so`/`libgemm.so` — 编译的 DCU 二进制核

### 最优权重/
- `student_fp16_10MB.pth` — **★最终 fp16 学生权重,推理即用**(来自 提交_pangu_weather/_stage/checkpoints/student.pth)
- 其它阶段权重在 `../../结果备份/`(636MB,student_mom99.pth 23MB / student_scnet_e96_p8.pth 等,对标参考非最终)

### 复赛课程/(官方赛题分讲 + 最优项目代码,教学复现用)
- 讲义 HTML:`00_课程总览` / `L1_推理部署` / `L2_matrix_core_HIP` / `L3_训练_真值监督蒸馏` / `L4_评测打分`
- `最优项目代码/03_训练/distill_truth.py` — **完整蒸馏实现**(EMA + 梯度检查点 + 双指标早停,38KB 最详版)
- `最优项目代码/02_matrix_core_HIP/` — DCU GEMM 内核(gemm_lib_mmac.hip / gemm_wint8.hip 等)
- `最优项目代码/04_评测/score15_dcu.py` — 官方 15 指标 DCU 评测

### 官方参考/(理解教师结构)
- `pangu.py` + `pangu_*.py` — 官方 Pangu 教师模型定义;`era5_new.py` — ERA5 数据集处理

### 先导杯气象大模型赛题_总结与思路.html — 竞赛经验总结

## 服务器最优文件(大文件不在本地)
- **训练机 mitai** 4×4090:`/data2/sunkai/`(era5_full 829 时刻);助手 `_mitai.py`,`distill_cache.py`(RAM 缓存 + bf16)
- **DeepLN** 云 V100:助手 `_deepln.py`;教师权重转换版 `model_bak_conv.pth`
- **funhpc** A100:`/data/era5/era5_full`(1900 时刻);**abc66** 5090:完整 ERA5
- **官方真实教师**:DCU 上 `model_bak.pth`(1.8G,真参数 64.22M,大头是 attn_mask buffer)
- 各服务器连接/env/数据目录速查见记忆 `servers-config-quickref`

## 复现路径
1. 部署:用 `最优代码/`,权重 `最优权重/student_fp16_10MB.pth`,跑 `python inference.py` → `python result.py`
2. 训练:参考 `复赛课程/最优项目代码/03_训练/distill_truth.py`(完整蒸馏),数据 era5_full,机器 mitai/DeepLN
3. 判优必须 eval 对真值 RMSE,别只看 val_match(见记忆 pangu-teacher-and-physics-prior)

## 关键结论/教训(见记忆)
- **减脂(训练时降维)才是低秩正解**,SVD 事后拆分否定(pangu-student-distill)
- **物理先验搬到架构层**(后置守恒校正硬约束)才涨点,软约束抢梯度失败(pangu-teacher-and-physics-prior)
- **真值监督破天花板**:温度掺真值近地面真超教师;但共享 recovery 头致高空漂移,修正=full-temp 全 13 层监督(pangu-truth-supervision)
- 量化:INT8 权重近无损但主帮模型大小;显存靠空间分块 tiling(pangu-quantization)
