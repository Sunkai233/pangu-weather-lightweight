# =============================================================================
# 文件: era5_new.py
# 作用: ERA5 数据管线——把磁盘上的 .h5 气象数据读成 (输入帧, 目标帧, 太阳天顶角, ...) 的样本
# 在流程中的位置: train.py / inference.py 通过它拿到 DataLoader,再喂给模型
# 关键点:
#   1) 一个样本 = 连续时间步:前 input_steps 帧当输入,后 output_steps 帧当预测目标(默认各 1 帧,即 t → t+6h)
#   2) channel_indices:按"变量名"从 h5 里挑出需要的 69 个通道(不是按前 69 个位置!顺序对齐靠名字)
#   3) 归一化在 __getitem__ 里做:(x - mu) / sd,mu/sd 来自 stats/global_means|stds.npy
#   4) 额外算了 cos_zenith(太阳天顶角余弦)作为物理辅助输入
# 注意: 本文件是"自包含版"数据管线(直接读 data_merged/{年}.h5);官方 inference.py 用的是
#       onescience 打包版 ERA5Datapipe(读 data/{年}/*.h5)。两者数据组织略不同,原理一致,可对照理解。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================

import os
import glob
import h5py                # 读写 HDF5(.h5)格式的库,气象大数据常用
import pytz                # 时区处理(这里用 UTC)
import numpy as np
import torch

from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# onescience 提供的两个气象工具:
from onescience.datapipes.climate.utils.invariant import latlon_grid          # 生成经纬度网格
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle  # 由时间+经纬度算太阳天顶角余弦


class ERA5Datapipe:
    # 数据管线封装:给定目录/年份/变量,吐出一个可迭代的 DataLoader。
    def __init__(
        self,
        dataset_dir,          # 数据根目录
        used_years,           # 用哪些年份(list[int])
        used_variables,       # 用哪些变量(list[str],即那 69 个通道名)
        distributed=False,    # 是否分布式(多卡)
        input_steps=1,        # 输入取几帧
        output_steps=1,       # 预测目标取几帧
        normalize=True,       # 是否归一化
        batch_size=1,
        num_workers=4,
    ):
        self.dataset_dir   = dataset_dir
        self.used_years    = used_years
        self.used_variables = used_variables
        self.distributed   = distributed
        self.input_steps   = input_steps
        self.output_steps  = output_steps
        self.normalize     = normalize
        self.batch_size    = batch_size
        self.num_workers   = num_workers


    def get_dataloader(self, mode):
        # 用上面的参数实例化底层 Dataset
        dataset = ERA5Dataset(
            dataset_dir=self.dataset_dir,
            used_years=self.used_years,
            used_variables=self.used_variables,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            normalize=self.normalize,
        )
        is_train = (mode == 'train')     # 训练模式才打乱数据

        # 分布式时用 DistributedSampler 把数据切给各张卡;单卡则为 None
        sampler = DistributedSampler(dataset, shuffle=is_train) if self.distributed else None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # 注意:用了 sampler 时不能再传 shuffle=True,所以这里 (训练 且 非分布式) 才 shuffle
            shuffle=(is_train and not self.distributed),
            sampler=sampler,
            drop_last=self.distributed,
        ), sampler


class ERA5Dataset(Dataset):
    # 真正的 Dataset:实现 __len__ 和 __getitem__,按索引返回一条样本。
    def __init__(
        self,
        dataset_dir,
        used_years,
        used_variables,
        mode='train',
        input_steps=1,
        output_steps=1,
        normalize=True,
    ):
        self.dataset_dir    = dataset_dir
        self.used_years     = used_years
        self.used_variables = used_variables
        self.input_steps    = input_steps
        self.output_steps   = output_steps
        self.normalize      = normalize

        # 初始化三件事:可用样本索引、归一化统计量、经纬度网格
        self._init_avail_samples()
        self._init_normalized_files()
        self._init_latlon_grid()


    def _init_avail_samples(self):
        # ① 找到 data_merged/ 下所有 {年}.h5 文件,解析出有哪些年份
        h5_files = sorted(glob.glob(os.path.join(self.dataset_dir, "data_merged", "*.h5")))
        available_years = [int(os.path.basename(f).replace(".h5", "")) for f in h5_files]

        # 校验:config 里要的年份是否都有数据,缺了就直接报错
        missing_years = [y for y in self.used_years if y not in available_years]
        if missing_years:
            raise ValueError(f"❌ Years not found in dataset: {missing_years}")

        # ── 读取变量信息 & 校验 ───────────────────────────────
        # 打开第一个 h5,读出数据形状和它内部记录的变量名清单
        with h5py.File(h5_files[0], "r") as f:
            ds = f["fields"]                       # "fields" 是数据集主键,形状 [T, C, H, W]
            self.T, self.C, self.H, self.W = ds.shape   # T=时间步数, C=总通道数, H=721, W=1440
            # h5 属性里存了每个通道对应的变量名(可能是 bytes,统一 decode 成 str)
            all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
            self.time_step = int(ds.attrs["time_step"])  # 相邻时间步间隔(小时),一般是 6

        # 校验:要的变量是否都在 h5 里
        missing_vars = [v for v in self.used_variables if v not in all_variables]
        if missing_vars:
            raise ValueError(f"❌ Variables not found in dataset: {missing_vars}")

        # ── 建立索引 ──────────────────────────────────────────
        # ★关键:按"变量名"找出它们在 h5 通道里的位置。这样即使 h5 里通道更多/顺序不同,也能精确取到要的 69 个
        self.channel_indices = [all_variables.index(v) for v in self.used_variables]
        # 年份 → 该年 h5 文件路径 的映射
        self.file_map = {
            y: os.path.join(self.dataset_dir,"data_merged", f"{y}.h5")
            for y in self.used_years
        }
        # 每年能切出多少个样本:一条样本要占用 input+output 连续帧,所以是 T - input - output + 1
        self.samples_per_year = self.T - self.input_steps - self.output_steps + 1
        self.total_samples    = len(self.used_years) * self.samples_per_year   # 总样本数


    def _init_normalized_files(self):
        # 读归一化统计量:全局均值/标准差,形状 [1, C, 1, 1](每个通道一个标量,广播到全图)
        stats_dir = os.path.join(self.dataset_dir, "stats")
        mu  = np.load(os.path.join(stats_dir, "global_means.npy"))  # [1, C, 1, 1]
        std = np.load(os.path.join(stats_dir, "global_stds.npy"))
        # 只取需要的 69 个通道的 mu/sd,转成 float32 张量
        self.mu = torch.as_tensor(mu[:, self.channel_indices, :, :], dtype=torch.float32)
        self.sd = torch.as_tensor(std[:, self.channel_indices, :, :], dtype=torch.float32)


    def _init_latlon_grid(self):
        # ── 经纬度网格 ────────────────────────────────────────
        # 生成 (纬度网格, 经度网格),范围纬 90→-90、经 0→360,形状都是 [H, W]
        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=(self.H, self.W))
        # 堆叠成 [2, H, W] 的张量(第 0 维:0=纬度,1=经度),给天顶角计算用
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)


    def __len__(self):
        return self.total_samples     # DataLoader 靠它知道一共多少条样本


    def __getitem__(self, idx):
        # 把全局样本编号 idx 拆成 (第几年, 该年内第几个起始帧)
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year     = self.used_years[year_idx]

        # ── 读取连续时间步 ────────────────────────────────────
        # 从该年 h5 里切出 [step_idx, step_idx+input+output) 这几帧,形状 [T', C, H, W]
        with h5py.File(self.file_map[year], "r") as f:
            frames = f["fields"][step_idx: step_idx + self.input_steps + self.output_steps]  # [T, C, H, W]
            frames = frames[:, self.channel_indices, :, :]    # 只保留需要的 69 个通道

        data   = torch.as_tensor(frames, dtype=torch.float32)
        invar  = data[:self.input_steps]   # [input_steps,  C, H, W]  前几帧 = 输入
        outvar = data[self.input_steps:]   # [output_steps, C, H, W]  后几帧 = 预测目标

        # 归一化:每个通道减均值除标准差(mu/sd 会自动广播到 H,W)
        if self.normalize:
            invar  = (invar  - self.mu) / self.sd
            outvar = (outvar - self.mu) / self.sd

        # ── 太阳天顶角 ────────────────────────────────────────
        # 由"目标帧的绝对时间"算太阳天顶角余弦(反映昼夜/太阳高度,是很强的物理先验)
        start_time = datetime(year, 1, 1, tzinfo=pytz.utc)   # 该年 1 月 1 日 0 点(UTC)为基准
        timestamps = np.array([
            # 目标帧对应的绝对时间戳(秒):基准 + (起始帧 + input + 第 t 个输出帧) × 每步小时数
            (start_time + timedelta(hours=(step_idx + self.input_steps + t) * self.time_step)).timestamp()
            for t in range(self.output_steps)
        ])
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()  # [output_steps, H, W]

        # 记录每一帧(输入+输出)的时间字符串 "YYYYMMDDHH",inference 用它给输出文件命名
        time_index = [
            (start_time + timedelta(hours=(step_idx + t) * self.time_step)).strftime("%Y%m%d%H")
            for t in range(self.input_steps + self.output_steps)
        ]

        # 返回一条样本:输入帧、目标帧、天顶角、该年内起始帧号、时间戳列表
        # squeeze(0):当 input_steps=output_steps=1 时,把 [1,C,H,W] 压成 [C,H,W]
        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx, time_index
