# 决赛学习路线(16 单元 + baseline 借鉴对照)

面向先导杯气象决赛(闭卷、从原始模型开始)的**建模→训练→推理→评测**全链路学习资料。

## 目录
- **[`学习总表与借鉴对照.md`](学习总表与借鉴对照.md)** —— ★主文档:16 单元总表,逐单元标清「可从 baseline/成熟包直接借鉴(不用背)」vs「必须自己能默写的核心考点」+ 最高频考点清单。
- **[`训练与调试实战.md`](训练与调试实战.md)** —— ★怎么在环境里搭/训/调:6 大考点各一段可跑最小自测(可逆/形状/无NaN/正交)+ 端到端训练命令 + 健康信号 + 常见 bug 修法表。
- **[`selftest_6考点.py`](selftest_6考点.py)** —— 一键跑 6 考点 PASS/FAIL(**DCU 实测 6/6 全过**);`python selftest_6考点.py`。
- **[`onescience网络库参考.md`](onescience网络库参考.md)** —— ★决赛闭卷唯一能读的"网络写法"来源:onescience.modules 工厂/style 清单 + Pangu 五大积木核心写法 + 11 种注意力/FNO 谱卷积/各家气象模型可抄清单。配 `baseline/onescience源码/modules/`(257 个 .py 整包)。
- `baseline/官方源码/` —— 官方 baseline 全套代码(inference/result/train/maxvit3d_student/distill_*/conf/config.yaml),对照学习用。
- `baseline/onescience源码/` —— onescience 关键源码(Pangu 教师 / ERA5Datapipe / YParams;赛题容器自带包,源码即文档)。
- `baseline/权重与数据说明.md` —— 教师权重(1.87G)与 ERA5 数据(太大不入库)的容器位置与获取方式。

## 怎么用
1. 按 `学习总表` 逐单元学:先看"成熟包/API"和"baseline 可借鉴位置"理解结构,再动手默写"必须自己写的考点"。
2. 对着 `baseline/官方源码/maxvit3d_student.py`(学生全部组件)+ `train.py`(训练)+ `inference.py`/`result.py`(推理评测)印证。
3. 成品实现见本仓库根 `最优代码/`;优化实测见 finals-toolkit 仓的 playbook。

进度:✅#1–4 已完成 · ⏳#5 下一步 · #6–16 待学习。
