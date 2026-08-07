# =============================================================================
# 文件: inference.py
# 作用: 官方推理脚本——加载权重,对测试集逐样本预测,存结果 npy + 记录每次 model() 耗时
# 在流程中的位置: train.py 训练出权重后 → inference.py 出预测 → result.py 算 RMSE/ACC
# 关键点(★决赛必记):
#   1) ★计时区只包住 model(invar) 这一行(start/end 之间),被 "AI4S(时间度量,位置不可更改)" 框住,
#      任何优化都不能把计算挪出这个区间,也不能改这三行的位置。评分的"推理时长 V"就来自 time_record.json。
#   2) 模型输入是 72 通道:invar[:, :4](4 地面) + surface_mask(3 静态) + invar[:, 4:](65 高空)拼接而成。
#   3) 输出反归一化:pred * std + means,还原成物理量再存;评分在物理量上算。
#   4) 输出目录 './result/output/' 不可更改,文件名 = 该样本时间戳(如 2012071606.npy)。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================

import torch
import os
import sys
import glob
import numpy as np
import h5py
from tqdm import tqdm
import time
import json
from onescience.models.pangu import Pangu            # 官方 Pangu 模型(教师/大模型)
from onescience.utils.YParams import YParams         # 读 config.yaml 的工具
from onescience.datapipes.climate import ERA5Datapipe  # 官方打包版数据管线


def get_stats(data_dir, channels):
    """从 metadata.json 读取变量列表，提取归一化参数"""
    # metadata.json 里存了全部变量名的顺序
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]

    # ★按"变量名"定位通道下标(和数据管线里一致,保证顺序对齐),而不是简单取前 N 个
    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))   # [1, C, 1, 1]
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]     # 只取需要通道的均值
    stds = std[:, channel_indices, :, :]     # 只取需要通道的标准差
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)            # 把当前目录加进 import 搜索路径

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")       # 读 "model" 段(超参)
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")  # 读 "datapipe" 段(数据配置)

    # 取归一化统计量(反归一化时要用)
    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    # 构建测试集 DataLoader
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    # ── 读取 3 个静态场(不随时间变,作为额外输入通道)──
    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))   # 陆海掩码
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))   # 土壤类型
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32)) # 地形高度
    # 地形做标准化(减均值除标准差),量纲太大会影响训练
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    # 把 3 个静态场堆成 [3, H, W] 放到 GPU
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    # 扩成 [B, 3, H, W](按 batch 复制),后面和每个样本拼接
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    # ── 加载模型权重 ──
    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")   # 官方基线权重
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])   # 把权重灌进模型

    model.eval()                                       # 切到评估模式(关 dropout 等)
    os.makedirs('result/output/', exist_ok=True)                          # AI4S, 输出路径不可更改
    print(f"📂 samples will be generated to './result/output/'")

    time_list = []       # 记录每个样本 model() 的耗时(秒),最后写 time_record.json
    first = True
    with torch.no_grad():                              # 推理不需要梯度,省显存加速
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]        # 输入帧 [B, 69, H, W](已归一化)
            outvar = data[1]       # 目标帧(这里推理不用,评分在 result.py 里另读标签)
            filename = data[4][-1][0]   # 取"最后一帧(目标帧)"的时间戳字符串,做输出文件名
            if first:
                first = False
                print(f"  invar  shape: {list(invar.shape)}   ← [batch, channels, H, W]")
                print(f"  outvar shape: {list(outvar.shape)}  ← [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            # ── 拼出 72 通道输入 ──
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float32)    # 前 4 个 = 地面变量
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)  # 后 65 个 = 高空变量
            # 拼接顺序 = [4 地面 | 3 静态掩码 | 65 高空] = 72 通道,正好是 Pangu.forward 期望的输入
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            #----------------------AI4S(时间度量不可更改)---------------------------
            start_time = time.perf_counter()      # AI4S(时间度量，位置不可更改)  ← 计时开始
            out_surface, out_upper_air = model(invar)   # ★唯一被计时的一行:模型前向。评分 V 就量这里
            end_time = time.perf_counter()        # AI4S(时间度量，位置不可更改)  ← 计时结束
            time_list.append(end_time-start_time) # AI4S(时间度量，位置不可更改)  ← 累计这次耗时
            #---------------------------------------------------------------------

            # 模型输出:out_surface [B,4,H,W] + out_upper_air(高空,需 reshape 回 65 通道)
            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)   # [B,65,H,W]
            # 拼成完整 69 通道预测,搬回 CPU 转 numpy
            pred_var = torch.concat([out_surface, out_upper_air], dim=1).cpu().numpy()
            # ★反归一化:还原成物理量(评分在物理量上做)
            pred_var = pred_var * stds + means
            # 存成 npy,文件名 = 时间戳(result.py 靠它和标签配对)
            np.save(f"result/output/{filename}.npy", pred_var)


        #----------------------AI4S(时间度量不可更改)---------------------------
        # 保存到 time_list.json 文件(评分脚本据此算"推理时长 V")
        with open("result/time_record.json", "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
        #---------------------------------------------------------------------
