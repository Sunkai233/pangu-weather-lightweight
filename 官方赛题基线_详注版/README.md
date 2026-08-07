# 官方赛题基线 · 详注版

> **这是什么**：先导杯气象大模型轻量化赛题**最初让拉取的那套官方基线代码**（独立版 Pangu-Weather
> 源码 + ERA5 数据管线 + 推理/评测脚本），我把它整理成一个**独立、自包含、逐行加了详细中文注释**的
> 文件夹，另配两份说明文档。用于决赛闭卷前吃透"数据怎么读、模型怎么搭、推理怎么计时、成绩怎么算"。
>
> **重要**：所有 `.py` / `.yaml` 都经过 **AST/取值校验，代码零改动，只新增了中文注释**（详见下方"可信度"）。

---

## 一、先看两份文档

- **[`数据说明.md`](数据说明.md)** —— ★数据专题：目录结构、69 通道明细、h5 格式、归一化、静态场、72 通道拼接、通道对齐陷阱。**先读这份。**
- 本 `README.md` —— 整体流程、文件导航、怎么跑、评分口径。

---

## 二、整体流程（一图串起来）

```
config.yaml ─┬─► train.py     训练大模型，存权重 model_bak.pth + loss 曲线
             │
 ERA5 数据 ──┤   （数据管线：era5_new.py / onescience ERA5Datapipe）
             │
             ├─► inference.py  加载权重 → 拼 72 通道 → model() 逐样本预测
             │                 存 result/output/{时间戳}.npy + result/time_record.json（★计时）
             │
             └─► result.py     预测 vs 真值，逐通道算 RMSE + ACC，打印成绩表
```

**三项得分**（总分 = U + V + W）：
- **U 轻量化**：模型大小 + 采样显存（本文件夹不涉及，见项目其他材料）。
- **V 推理时长**：来自 `time_record.json`，**只计时 `inference.py` 里 `model(invar)` 那一行**（被 `AI4S(时间度量,位置不可更改)` 框住，三行位置不能动）。
- **W 预测性能**：来自 `result.py` 的每通道 RMSE（越小越好）+ ACC 距平相关（越接近 1 越好）。

---

## 三、文件导航（都带详细中文注释）

**核心管线（我逐行精注，重点看这 5 个）：**
- `config.yaml` —— 总配置：模型超参 + 数据 + 归一化 + DataLoader；标了哪些是赛题硬约束。
- `era5_new.py` —— ERA5 数据管线：怎么从 h5 读样本、按名字取 69 通道、归一化、算太阳天顶角。
- `inference.py` —— ★推理 + 计时区 + 72 通道拼接 + 反归一化 + 存盘。
- `result.py` —— ★评分：RMSE / ACC 的算法与通道对齐。
- `pangu.py` —— ★主模型：5 大积木按 U 形串联（embed→layer1→down→layer2/3→up→layer4→recovery + skip）。

**模型积木（Pangu 五大件的独立实现，带注释）：**
- `panguembedding2d.py` / `panguembedding3d.py` —— Patch Embedding：`ZeroPad + Conv(stride=patch)` 切 patch。
- `pangufuser.py` —— 主干融合块：堆叠 `EarthTransformer3DBlock`，偶/奇层普通/移窗交替。
- `pangupatchrecovery2d.py` / `pangupatchrecovery3d.py` —— Patch Recovery：`ConvTranspose` 反卷积恢复 + center-crop。
- `pangu_layer.py` —— `Mlp` / `DropPath`（前馈网络与随机深度）。
- `pangu_utils.py` —— 窗口划分/恢复、padding/裁剪、地球位置索引、移窗掩码等工具（原文件已带较完整 docstring）。

**其它：**
- `metadata.json` —— 69 个变量名的权威顺序（原样保留，JSON 不能写注释，其含义见 `数据说明.md` 第六节）。

> 注：`pangu.py` 里 `OneEmbedding/OneFuser/OneSample/OneRecovery` 是 onescience 的**工厂**，
> `style="PanguXXX"` 会映射到本文件夹这几个积木的**同名实现**。也就是说：读 `pangu.py` 看整体怎么串，
> 读这几个积木文件看每一件具体怎么写。

---

## 四、怎么跑（赛题环境内）

```bash
source ../earth_env.sh          # 激活赛题环境

# 1) 训练（单卡）
python train.py

# 2) 推理 → 结果输出到 ./result/output/
python inference.py

# 3) 评测 & 出图
python result.py
```

无真实数据时可先用伪造数据验证流程（把 `config.yaml` 的 `max_epoch` 设为 1）：
```bash
python fake_data.py             # 伪造 ERA5 数据，仅验证代码能跑通
```

> 提交前务必把 `config.yaml: test_ratio` 改成 `[2050, 2052, 2054, 2056, 2058]`（后台真实测试的“标识年”）。

---

## 五、可信度（代码零改动的证明）

本文件夹的注释版是在官方原始文件基础上**只新增中文注释**得到的：
- 4 个核心 `.py`（era5_new/inference/result/pangu）用 **Python AST 对比**原文件，结果**完全一致**（注释不影响 AST，AST 一致 ⇒ 代码逻辑逐字节等价）。
- `config.yaml` 用 `yaml.safe_load` 对比原文件，**取值完全一致**。
- 积木文件同样只加注释、保留原有 docstring 与代码。

原始未改动的文件仍在仓库顶层 `官方参考/`，可随时对拍。
