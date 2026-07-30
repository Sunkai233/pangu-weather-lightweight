# -*- coding: utf-8 -*-
"""生成训练用配置 conf/config_train.yaml —— 接入共享 ERA5 数据集。

为什么要单独一份配置:
  提交包里的 conf/config.yaml 是【推理/评测】配置(test_ratio = 2050~2058),评测机按它读数据、
  写结果,改动它会直接导致评测失败。训练需要的是另一批年份(1980~2000)和另一个数据根目录,
  因此派生出独立的 config_train.yaml,原文件一个字都不动。

共享数据集结构(官方问答给的路径):
  /public/SothisAI/sharingCenter/xiaodao2026/ERA5/old-data/
    ├── data/1980/1980010100.h5 ...   每 6 小时一个文件(393MB),1980~2000 共 21 年
    ├── stats/                        各变量 means/stds(归一化用)
    ├── static/                       land_mask / soil_type / topography
    └── metadata.json

注意 era5_old.py 里 train_ratio / val_ratio 这两个字段名有误导性 —— 它们不是"比例",
而是【年份列表】(源码:selected_years = getattr(params, f"{mode}_ratio"))。

用法(在 pangu_weather/ 下):
  python make_config_train.py                       # 默认 1980~1996 训练, 1997~1998 验证
  python make_config_train.py --train 1980-1990 --val 1997
"""
import argparse, os, shutil, sys
import yaml

SHARED = "/public/SothisAI/sharingCenter/xiaodao2026/ERA5/old-data"
# 备用路径(官方问答里给过的旧位置,某些容器挂的是这个)
SHARED_ALT = "/work2/share/sugonhpcapp01/ERA5/old-data"


def parse_years(s):
    """'1980-1996' 或 '1980,1981,1985' → [int]"""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="conf/config.yaml", help="推理配置(只读,不修改)")
    ap.add_argument("--dst", default="conf/config_train.yaml")
    ap.add_argument("--root", default="", help="共享数据根目录(留空则自动探测)")
    ap.add_argument("--train", default="1980-1996", help="训练年份")
    ap.add_argument("--val", default="1997,1998", help="验证年份")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader 并行读盘进程数。★这是训练速度的头号开关:"
                         "赛方默认 num_workers=0(单线程串行读),而每个样本是一个 411MB 的 h5、"
                         "单线程 NFS 只有 200~300MB/s —— 实测一个 iter 里读盘要 4.4~6.1 秒,"
                         "而教师前向才 0.8 秒、学生前反向才 0.24 秒,GPU 有 85% 的时间在纯等盘。"
                         "开 8 个 worker 并行读 + 预取重叠后,读盘时间被算力掩盖,每样本降到约 1 秒")
    args = ap.parse_args()

    root = args.root
    if not root:
        for cand in (SHARED, SHARED_ALT):
            if os.path.isdir(os.path.join(cand, "data")):
                root = cand
                break
    if not root:
        sys.exit(f"未找到共享数据集,请用 --root 指定。试过:\n  {SHARED}\n  {SHARED_ALT}")

    have = sorted(int(d) for d in os.listdir(os.path.join(root, "data")) if d.isdigit())
    tr = [y for y in parse_years(args.train) if y in have]
    va = [y for y in parse_years(args.val) if y in have]
    miss = [y for y in parse_years(args.train) + parse_years(args.val) if y not in have]
    if not tr or not va:
        sys.exit(f"训练/验证年份为空。数据集里可用年份: {have}")

    with open(args.src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["datapipe"]["dataset"]
    ds["data_dir"] = root                       # datapipe 会自己进 data/<年> 找 .h5
    ds["stats_dir"] = os.path.join(root, "stats")
    ds["static_dir"] = os.path.join(root, "static")
    ds["train_ratio"] = tr                      # 字段名叫 ratio,实为年份列表
    ds["val_ratio"] = va
    # test_ratio 保留原样:训练用不到,留着也无害

    # ★ 并行读盘:训练速度的头号开关(实测读盘占一个 iter 的 85%)
    dl = cfg["datapipe"]["dataloader"]
    old_w = dl.get("num_workers", 0)
    dl["num_workers"] = args.workers
    dl["pin_memory"] = True

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[写出] {args.dst}")
    print(f"  data_dir  : {root}")
    print(f"  可用年份  : {have[0]}~{have[-1]} (共 {len(have)} 年)")
    print(f"  训练年份  : {tr[0]}~{tr[-1]} ({len(tr)} 年)")
    print(f"  验证年份  : {va}")
    if miss:
        print(f"  跳过(数据集中没有): {miss}")
    print(f"  通道数    : {len(ds['channels'])}  批大小: {dl['batch_size']}")
    print(f"  ★ num_workers: {old_w} → {args.workers}  (并行读盘;实测单 iter 里读盘 4.4~6.1s、"
          f"教师仅 0.8s,GPU 85% 时间在等盘)")
    print(f"\n  ★ conf/config.yaml(推理/评测配置)未作任何改动")


if __name__ == "__main__":
    main()
