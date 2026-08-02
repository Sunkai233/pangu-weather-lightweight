# 决赛学习路线(16 单元 + baseline 借鉴对照)

面向先导杯气象决赛(闭卷、从原始模型开始)的**建模→训练→推理→评测**全链路学习资料。

## 目录
- **[`学习总表与借鉴对照.md`](学习总表与借鉴对照.md)** —— ★主文档:16 单元总表,逐单元标清「可从 baseline/成熟包直接借鉴(不用背)」vs「必须自己能默写的核心考点」+ 最高频考点清单。
- `baseline/官方源码/` —— 官方 baseline 全套代码(inference/result/train/maxvit3d_student/distill_*/conf/config.yaml),对照学习用。
- `baseline/onescience源码/` —— onescience 关键源码(Pangu 教师 / ERA5Datapipe / YParams;赛题容器自带包,源码即文档)。
- `baseline/权重与数据说明.md` —— 教师权重(1.87G)与 ERA5 数据(太大不入库)的容器位置与获取方式。

## 怎么用
1. 按 `学习总表` 逐单元学:先看"成熟包/API"和"baseline 可借鉴位置"理解结构,再动手默写"必须自己写的考点"。
2. 对着 `baseline/官方源码/maxvit3d_student.py`(学生全部组件)+ `train.py`(训练)+ `inference.py`/`result.py`(推理评测)印证。
3. 成品实现见本仓库根 `最优代码/`;优化实测见 finals-toolkit 仓的 playbook。

进度:✅#1–4 已完成 · ⏳#5 下一步 · #6–16 待学习。
