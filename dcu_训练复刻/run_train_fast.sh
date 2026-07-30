#!/bin/bash
# 蒸馏训练(直接读 NFS 上的 int8 缓存)。反复执行即自动从 .ckpt 续训,不会从头。
#
# 为什么不拷到本地盘:
#   本地盘随机读 664MB/s、NFS 只有 73MB/s,每轮 330s vs 803s。但把 183GB 从 NFS
#   冷读回本地盘要 40+ 分钟,而且实测会把容器的 IO 压到连 shell 都进不去(直接挂掉)。
#   本平台的容器会被定期回收(实测一天两次),每次回收后本地盘清空、都得重拷一遍。
#   慢 2.4 倍换取"回收后能立刻续训、不会压死容器"——这笔账划算。
#
# worker 调到 6:NFS 随机读慢,靠更多并发预取来补(每个 worker 读 216MB/0.05s→NFS 上约 3s)。
cd /public/home/xdzs2026_c296/pangu_weather
export PYTHONDONTWRITEBYTECODE=1
nohup python -u distill_dcu.py --config conf/config_train.yaml \
  --cache-dir /public/home/xdzs2026_c296/cache_int8 --loader-workers 6 \
  --embed 96 --depths 2,4,2 --heads 6,12,6 --patch 2,16,16 \
  --global-mode grid --mlp-ratio 2.0 \
  --opt adamw --lr 6e-4 --warmup 5 --epochs 60 \
  --alpha 0.5 --aug 1 --spec 0 \
  --save /public/home/xdzs2026_c296/student_dcu_repro.pth \
  >> /public/home/xdzs2026_c296/train_dcu.log 2>&1 &
echo "started pid=$!"
