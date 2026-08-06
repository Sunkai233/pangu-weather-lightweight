# 实操 notebooks(7 个知识点 · Jupyter 一格一格跑)

覆盖决赛剩余核心链路:**真实数据 → 单Batch反向 → 训练循环 → checkpoint → 推理 → RMSE/ACC评测 → 提交包**。
每个 notebook 顶部一格 `SETUP`(含绝对路径 `BASE`),之后每格独立可跑;关键处带 `assert` 自测。**全部代码已在海光 DCU 实测**。

## 用法
1. 在赛题 DCU 容器的 Jupyter 里打开(容器自带 onescience/torch)。
2. **先改第一格的 `BASE`** 为你的主目录(默认 `/public/home/xdzs2026_c296`)。
3. 每格从上往下依次运行(`Shift+Enter`),看输出/断言。报错就停在那格查 `shape/dtype/isnan`。

## 目录(按比赛重要性 = 学习顺序)
| notebook | 学什么 | 关键考点 |
|---|---|---|
| `01_真实数据与配置读取.ipynb` | YParams / ERA5Datapipe / 真实batch / 静态场 / **99→69按名字** / 6h配对 | data[0]=已归一69ch;sel=按metadata名字(非前69) |
| `02_单Batch真正反向更新.ipynb` | zero_grad/backward/step + 6项验证 | 学生有梯度·教师梯度None·学生权重变·教师不变·loss有限 |
| `03_标准训练循环.ipynb` | epoch/batch循环 / bf16 / warmup+cosine / 裁剪 / 存最优 / 日志 | 只存val最优;教师输出可缓存 |
| `04_Checkpoint保存与恢复.ipynb` | 存完整字典 / 续训 / 推理加载 / module前缀 / config匹配 | 存最优非最后一轮;strict=True验证 |
| `05_标准推理流程.ipynb` | half/eval/inference_mode / 拼72通道 / 反归一化 / 存npy | ★计时只包model(x);输入72出69;存前反归一化 |
| `06_官方RMSE与ACC评测.ipynb` | 每通道RMSE + 距平相关ACC + 平均 | ★99→69按名字;预测与标签同config顺序 |
| `07_最终提交包与无网自测.ipynb` | 包结构 / 无网import / **相对路径** / 输出目录 / 打包 | 提交用相对路径;计时区纯净;顶层一个目录无pycache |

## 对照
- 完整可跑成品脚本(int8缓存加速的训练、修正版评测)见上一级目录:`fast_distill3.py`、`inference_val.py`、`result_fixed.py`。
- 6大必背考点最小自测另见:`../训练与调试实战.md` + `../selftest_6考点.py`。
- baseline 源码对照:`../baseline/官方源码/`(maxvit3d_student.py 等)。

> 实测记录:NB2 在 DCU 逐格执行,6 项断言全过;数据/推理/评测的底层代码由 `ground_*` 测试与前期完整训练验证。
