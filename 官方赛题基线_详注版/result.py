# =============================================================================
# 文件: result.py
# 作用: 官方评分脚本——把 inference 存的预测 npy 和真值 h5 逐通道比对,算 RMSE 和 ACC
# 在流程中的位置: inference.py 出预测后运行它,得到每通道的 RMSE/ACC(评分 W 的来源)
# 关键点(★决赛必记):
#   1) RMSE(均方根误差):越小越好,直接量预测和真值差多少。
#   2) ACC(距平相关系数 Anomaly Correlation):把预测和真值都减去气候平均 clim_mean(=global_means),
#      再算相关系数,衡量"异常/距平"预报得准不准,越接近 1 越好。
#   3) ★通道对齐陷阱:本官方版用 channel_indices = range(len(channels)) 直接取前 N 个通道;
#      若真值 h5 通道数 != 预测通道数或顺序不同,会错位!(我们复现时改成"按变量名取"的 result_fixed.py 更稳)。
#   4) clim_mean 用 global_means(全局气候平均)当作距平基准。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import h5py
from datetime import datetime
from tqdm import tqdm
from onescience.utils.fcn.YParams import YParams
from matplotlib import rcParams

# ── matplotlib 画图样式(不影响评分,只影响出图美观)──
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9

def get_metadata(data_dir, test_years, channels):
    """Multi-file HDF5 format: data/{year}/*.h5"""
    # 收集测试年份下所有真值 h5(每个时间戳一个文件)
    h5_files = []
    for year in test_years:
        year_files = sorted(glob.glob(os.path.join(data_dir, "data", str(year), "*.h5")))
        h5_files.extend(year_files)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}/data/{{year}}/")
    # ★这里直接用前 len(channels) 个通道下标(假设真值 h5 通道顺序与 channels 一致)
    channel_indices = list(range(len(channels)))

    # 列出 inference 生成的全部预测 npy(按文件名排序)
    total_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    total_files.sort()

    # Build file map: YYYYMMDDHH → HDF5 path   建"时间戳 → 真值 h5 路径"映射
    h5_map = {}
    for h5f in h5_files:
        basename = os.path.basename(h5f).replace('.h5', '')
        h5_map[basename] = h5f
    return total_files, channel_indices, h5_map

def get_result(total_files, channel_indices, h5_map, clim_mean):
    # 逐通道累加器
    channel_rmse = np.zeros(len(channel_indices))
    channel_acc = np.zeros(len(channel_indices))
    clim_mean = clim_mean[0, :, :, :]              # [C, H, W] 气候平均(距平基准)
    numerator = np.zeros(len(channel_indices))     # ACC 分子:Σ(预测距平 · 真值距平)
    pred_sq_sum = np.zeros(len(channel_indices))   # ACC 分母项:Σ(预测距平²)
    label_sq_sum = np.zeros(len(channel_indices))  # ACC 分母项:Σ(真值距平²)
    for file in tqdm(total_files, unit="files"):
        fname = file[:-4]  # 去掉 .npy               # 得到时间戳
        h5_path = h5_map[fname]                       # 找到对应真值 h5
        with h5py.File(h5_path, "r") as f:
            label = f["fields"][:].squeeze()  # [C, H, W]   读真值
            label = label[channel_indices]              # 取需要的通道
        pred = np.load(f'result/output/{file}').squeeze()  # 读预测

        # 距平 = 原值 - 气候平均
        label_anom = label - clim_mean
        pred_anom = pred - clim_mean
        # 累加(在 H,W 两个空间维求和)
        numerator += np.sum(pred_anom * label_anom, axis=(1, 2))
        pred_sq_sum += np.sum(pred_anom ** 2, axis=(1, 2))
        label_sq_sum += np.sum(label_anom ** 2, axis=(1, 2))

        # RMSE:每个文件先算 sqrt(mean((真值-预测)²)),再累加,最后除文件数取平均
        channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))
    channel_rmse /= len(total_files)                          # RMSE 对样本取平均
    # ACC = Σ(预测距平·真值距平) / sqrt(Σ预测距平² · Σ真值距平²)
    channel_acc = numerator / (np.sqrt(pred_sq_sum * label_sq_sum) + 1e-8)
    np.save('./result/acc.npy', channel_acc)                 # 存每通道 ACC
    np.save('./result/rmse.npy', channel_rmse)               # 存每通道 RMSE


def show_result():
    # 读回上一步存的每通道指标,打印成表
    channel_rmse = np.load('./result/rmse.npy')
    channel_acc = np.load('./result/acc.npy')

    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    w = 24  # 最长 channel 名宽度

    # 表头
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'RMSE':>12} │ {'ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    # 数据行:逐通道打印 RMSE / ACC
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {channel_rmse[i]:>12.4f} | {channel_acc[i]:>12.4f} |")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    # 最后一行:所有通道的平均 RMSE / 平均 ACC(整体成绩)
    print(f"│ {'Average':<{w}} │ {np.mean(channel_rmse):>12.4f} │ {np.mean(channel_acc):>12.4f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┘")

def plot(label, pred, var, filename):
    # 画"真值 / 预测 / 差值"三联图(可视化用,不影响评分)
    # 基础设置
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 坐标轴标签(经纬度)
    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    # 计算统一色条范围(真值和预测共用一个色标便于对比)
    vmin = min(label.min(), pred.min())
    vmax = max(label.max(), pred.max())

    # 计算差异和 RMSE
    diff = label - pred
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()

    # 三张子图的绘图配置
    plot_configs = [
        {'data': label, 'title': 'Truth', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': pred,  'title': 'Prediction', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': diff,  'title': f'Difference (RMSE={rmse:.2f})', 'cmap': 'RdBu_r', 'vmin': -diff_abs_max, 'vmax': diff_abs_max},
    ]

    # 统一绘制
    for ax, cfg in zip(axes, plot_configs):
        im = ax.imshow(cfg['data'], cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'])
        ax.set_title(cfg['title'], fontsize=12, pad=4)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        plt.colorbar(im, ax=ax, orientation='horizontal')

    # 总标题
    fig.suptitle(var, fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(train_loss, valid_loss):
    # 画训练/验证 loss 下降曲线(诊断训练用)
    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))   # 过滤掉 NaN 轮次
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    # 配置
    colors = {'train': '#2563EB', 'valid': '#EA580C'}
    epochs = np.arange(1, len(train_loss) + 1)

    # 绘制两条曲线
    ax.plot(epochs, train_loss, color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, valid_loss, color=colors['valid'], linewidth=1.5, label='Valid', linestyle='--')
    # 标注验证 loss 最小点(最佳 epoch)
    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx],
               color=colors['valid'], s=40, zorder=5, edgecolors='white')
    ax.annotate(f'Best: {valid_loss[min_idx]:.3f}',
                xy=(epochs[min_idx], valid_loss[min_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color=colors['valid'],
                arrowprops=dict(arrowstyle='-', color=colors['valid'], lw=0.5))

    # 坐标轴
    ax.set(xlabel='Epoch', ylabel='Loss', xlim=(0, len(train_loss) + 1))

    # 样式
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('./result/loss.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    # 先画 loss 曲线(读训练时存的 trloss.npy / valoss.npy)
    train_loss = np.load('./data/checkpoints/trloss.npy')
    valid_loss = np.load('./data/checkpoints/valoss.npy')
    plot_loss(train_loss, valid_loss)

    data_dir = cfg_data.dataset.data_dir
    test_years = cfg_data.dataset.test_ratio
    # 建立预测文件列表、通道下标、真值映射
    total_files, channel_indices, h5_map = get_metadata(data_dir, test_years, cfg_data.dataset.channels)

    # Load data & Compute RMSE/ACC per channel
    # 气候平均基准:直接用全局均值 global_means 当 clim_mean
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))

    clim_mean = mu[:, channel_indices, :, :]
    get_result(total_files, channel_indices, h5_map, clim_mean)   # 算并存 RMSE/ACC
    show_result()                                                  # 打印成绩表


    ##### 默认绘制 test_time 第一年的第一个时间步，用户可自行指定日期和变量 #####
    # (下面这段可视化是可选的,默认注释掉;想看某天某变量的三联图时取消注释并指定)
    # test_year = cfg_data.dataset.test_time[0]
    # eg_files = [f'2012071606']
    # channel_index = [cfg_data.dataset.channels.index(v) for v in ['2m_temperature', 'geopotential_500', 'temperature_500']]

    # selected_var = [cfg_data.dataset.channels[int(i)] for i in channel_index]
    # print(f"seleted date: {eg_files}")
    # print(f"selected channels: {selected_var}")
    # for file in eg_files:
    #     h5_path = h5_map.get(file)
    #     if h5_path is None:
    #         print(f"⚠️  No HDF5 file for {file}, skipping plot")
    #         continue
    #     with h5py.File(h5_path, "r") as f:
    #         label = f["fields"][:].squeeze()  # [C, H, W]
    #         label = label[channel_indices]
    #     pred = np.load(f'result/output/{file}.npy').squeeze()
    #     for i in range(len(selected_var)):
    #         filename = f'./result/{file}_{selected_var[i]}.png'
    #         plot(label[channel_index[i]], pred[channel_index[i]], selected_var[i], filename)
    #         print(f'✅plot {filename}')
