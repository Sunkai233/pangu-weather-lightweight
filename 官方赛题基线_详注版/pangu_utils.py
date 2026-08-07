# =============================================================================
# 文件: pangu_utils.py
# 作用: 工具函数大礼包 —— 窗口注意力需要的一切辅助逻辑: 补零对齐(get_pad)、裁边(crop)、
#       窗口划分/还原(window_partition / window_reverse)、地球位置索引(get_earth_position_index)、
#       移窗注意力掩码(get_shift_window_mask),外加 Mlp / DropPath / 权重初始化等。
# 在网络中的位置: 被主干每个 Transformer 块反复调用,是"看晕人"的形状变换都藏在这里。
# 关键点(务必看懂):
#   1) window_partition/window_reverse 是一对逆操作: 把 (B, 气压, 纬, 经, C) 切成一堆小窗口再拼回来。
#      经度方向(Lon)被特殊地并进 batch 维,因为地球经度是环状且位置偏置沿经度平移不变。
#   2) get_earth_position_index: 为"地球专属位置偏置"造一张查表索引 —— 偏置依赖 query/key 的绝对气压层、
#      绝对纬度,以及二者的相对经度,把 3 个坐标编码成 1 个整数索引去查可学习的偏置表。
#   3) get_shift_window_mask: 移窗(SW-MSA)后,原本不相邻却被 roll 卷到一起的格点不能互相注意,用 -100 掩掉。
#   4) 所有 get_pad/crop 都遵循同一规则: 补齐到能被窗口整除、前后对半补/裁。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
from torch import nn
import torch
import math


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        # 输出维度默认等于输入维度(残差相加要求进出同维)
        out_features = out_features or in_features
        # 隐藏层维度默认等于输入维度(实际常传入 in_features*mlp_ratio)
        hidden_features = hidden_features or in_features
        # 前馈网络: 两层线性 + 中间激活
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        # 升维 -> 激活 -> dropout -> 降维 -> dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """改编自 timm master
    按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）（当应用于残差块的主路径时）
    这与我为 EfficientNet 等网络实现的 DropConnect 功能相同，但原来的命名容易引起误解，因为 “Drop Connect” 
    在另一篇论文中指的是不同形式的 Dropout，参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956，
    选择将层和参数名称改为 Drop Path，而不是沿用 DropConnect 作为层名，同时将参数命名为 survival rate（存活率）。

    """
    # 丢弃概率为 0 或不在训练时,原样返回(推理不丢)
    if drop_prob == 0.0 or not training:
        return x
    # 存活概率
    keep_prob = 1 - drop_prob
    # 构造只在 batch 维上变化、其余维全为 1 的形状,保证"整条样本"一起被留/丢
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    # 按 keep_prob 做伯努利采样: 每个样本得到 0(丢)或 1(留)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # 除以存活率做缩放,保持输出期望不变(inverted dropout)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    # 逐样本相乘: 被丢的样本这一路残差整条清零
    return x * random_tensor


class DropPath(nn.Module):
    """摘自 timm 仓库
    按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）当应用于残差块的主路径时）
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        # 委托给上面的函数,把当前 training 状态传入
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"

def get_earth_position_index(window_size, ndim=3):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    构建位置索引（Position Index）以复用位置偏置的对称参数
    实现参考: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    参数:
        window_size (tuple[int]): 窗口大小，三维为 [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): 张量维度，2 表示二维窗口，3 表示三维窗口

    返回值:
        position_index (torch.Tensor): 位置索引矩阵，形状为：[win_pl * win_lat * win_lon, win_pl * win_lat * win_lon] or [win_lat * win_lon, win_lat * win_lon]
    """
    # 拆窗口尺寸: 三维含气压层,二维只含纬经
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
    elif ndim == 2:
        win_lat, win_lon = window_size

    if ndim == 3:
        # query 端的气压层坐标: 0,1,...,win_pl-1
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(win_pl)
        # key 端的气压层坐标: 乘以 -win_pl 做偏移编码(让 query/key 的绝对层号都能被区分)
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(win_pl) * win_pl

    # query 端的纬度坐标: 0,1,...,win_lat-1 (地球偏置依赖"绝对纬度",故 query/key 分别编码)
    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # key 端的纬度坐标: 乘以 -win_lat 做偏移编码
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # 经度只用一套坐标(偏置沿经度只看"相对"距离,平移不变)
    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # 用 meshgrid 把各维坐标铺成网格,再 stack 成 [坐标维=2或3, ...窗口各维...]
    # Change the order of the index to calculate the index in total
    if ndim == 3:
        # coords_1: query 每个格点的 (气压,纬,经) 坐标; 形状 [3, win_pl, win_lat, win_lon]
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        # coords_2: key 每个格点的坐标(气压/纬用了负偏移编码)
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    elif ndim == 2:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))
    # 把窗口内所有格点摊平: [坐标维, N],其中 N = 窗口内格点总数
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    # 做 query 与 key 的两两差: [坐标维, N_query, N_key],得到每对 (q,k) 在各维上的组合坐标
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    # 把坐标维挪到最后: [N_query, N_key, 坐标维]
    coords = coords.permute(1, 2, 0).contiguous()

    # 把每一维平移成从 0 开始,并乘上进位权重,准备把多维坐标压成一个整数索引
    # Shift the index for each dimension to start from 0
    if ndim == 3:
        # 经度维: 平移到 [0, 2*win_lon-2]
        coords[:, :, 2] += win_lon - 1
        # 纬度维: 乘经度跨度做进位
        coords[:, :, 1] *= 2 * win_lon - 1
        # 气压维: 乘上更高的进位权重
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    elif ndim == 2:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1

    # 三(二)维坐标求和 -> 每对 (q,k) 得到唯一整数索引,用来查可学习的位置偏置表
    # Sum up the indexes in two/three dimensions
    position_index = coords.sum(-1)

    return position_index

def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path
):
    # 若是 DataParallel/DDP 包装过的模型,取出内部真正的 model 再存(去掉 .module 前缀)
    model_to_save = model.module if hasattr(model, "module") else model
    # 打包模型权重 + 优化器 + 调度器 + 当前最优验证损失及其 epoch
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    # 存到指定目录下的固定文件名
    torch.save(state, f"{model_path}/pangu_weather.pth")

def get_pad3d(input_resolution, window_size):
    """
    参数:
        input_resolution (tuple[int]): 输入数据的分辨率 (Pl, Lat, Lon)，分别表示气压层、纬度和经度方向的大小。
        window_size (tuple[int]): 窗口大小 (Pl, Lat, Lon)，用于计算分块或滑动窗口时的尺寸。

    返回:
        padding (tuple[int]): 填充大小 (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)，用于保证输入数据可以整除窗口大小或适配滑动窗口。
    """
    # 拆出输入三维分辨率与窗口三维尺寸
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    # 六个方向补零量初始化为 0
    padding_left = (
        padding_right
    ) = padding_top = padding_bottom = padding_front = padding_back = 0
    # 各方向对窗口取余,判断是否需要补齐
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    # 气压层方向补齐(前后对半)
    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    # 纬度方向补齐(上下对半)
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    # 经度方向补齐(左右对半)
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    # 返回顺序: 左, 右, 上, 下, 前, 后
    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    参数:
        input_resolution (tuple[int]): 输入数据的分辨率 (Lat, Lon)，分别表示纬度和经度方向的大小。
        window_size (tuple[int]): 窗口大小 (Lat, Lon)，用于计算分块或滑动窗口时的尺寸

    返回:
        padding (tuple[int]): 填充大小 (padding_left, padding_right, padding_top, padding_bottom)，用于保证输入数据可以整除窗口大小或适配滑动窗口。
    """
    # 小技巧: 在最前面补一个假的气压层维(大小都设 2),直接复用 3D 版逻辑
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    # 只取前四个(左右上下),丢弃假气压维对应的前后补零
    return padding[:4]


def crop2d(x: torch.Tensor, resolution):
    """
    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, Lat, Lon)，其中 B 为批量大小，C 为通道数，Lat 和 Lon 分别为纬度和经度方向的尺寸。
        resolution (tuple[int]): 输入分辨率 (Lat, Lon)，对应纬度和经度方向的尺寸。
    """
    # 取当前(补齐后)的纬经尺寸
    _, _, Lat, Lon = x.shape
    # 算出比目标多出的量
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    # 上下对半裁
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    # 左右对半裁
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    # center-crop 回目标分辨率
    return x[
        :, :, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right
    ]


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): 输入张量，形状为 (B, C, Pl, Lat, Lon)，其中 B 为批量大小，C 为通道数，Pl 为气压层数量，Lat 和 Lon 分别为纬度和经度方向的尺寸。
        resolution (tuple[int]): 输入分辨率 (Pl, Lat, Lon)，对应气压层、纬度和经度方向的尺寸。
    """
    # 取当前(补齐后)的气压/纬/经尺寸
    _, _, Pl, Lat, Lon = x.shape
    # 各方向多出的量
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    # 气压方向前后对半裁
    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    # 纬度方向上下对半裁
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    # 经度方向左右对半裁
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    # center-crop 回目标三维分辨率
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        # 标准正态分布的累积分布函数 CDF
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # 若 mean 距离区间 [a,b] 太远,采样质量会很差,给出警告(注: 原文件未 import warnings,属遗留代码)
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 思路: 先在 [a,b] 对应的 CDF 区间上均匀采样,再用正态分布的反 CDF 变换回来
    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    # 区间下/上端在标准正态 CDF 下的值
    u1 = norm_cdf((a - mean) / std)
    u2 = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [u1, u2], then translate to
    # [2u1-1, 2u2-1].
    # 在 [2u1-1, 2u2-1] 上均匀填充(为下一步 erfinv 做准备)
    tensor.uniform_(2 * u1 - 1, 2 * u2 - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    # 反误差函数 = 反 CDF 变换,得到截断的标准正态样本
    tensor.erfinv_()

    # Transform to proper mean, std
    # 缩放到目标标准差、平移到目标均值
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    # 兜底裁剪,保证落在 [a,b] 内
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""从 timm 主分支中剪切粘贴而来。
    该函数使用截断正态分布为输入张量填充值。
    这些值实际上取自正态分布:math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    但若超出区间 [a, b]，则会重新采样，直到落入范围内为止。
    当满足 a ≤ mean ≤ b 时，该方法的效果最佳。:math:`a \leq \text{mean} \leq b`.

    NOTE: 注意：该实现与 PyTorch 的 trunc_normal_ 类似，
    但截断区间 [a, b] 是在对 mean 和 std 应用之后才生效，
    因此参数 a、b 应根据 mean 和 std 的范围进行调整。
    """
    # 权重初始化,no_grad 下原地填充,不记录计算图
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    参数:
        x: 输入张量，形状为 (B, Pl, Lat, Lon, C) 或 (B, Lat, Lon, C)。
        window_size (tuple[int]): 窗口大小，格式为 [win_pl, win_lat, win_lon] 或 [win_lat, win_lon]。
        ndim (int): 窗口的维度，取值为 3 或 2。

    返回:
        windows: 分块后的窗口张量，形状为(B × num_lon, num_pl × num_lat, win_pl, win_lat, win_lon, C) 或(B × num_lon, num_lat, win_lat, win_lon, C)。
    """
    # ===== 三维窗口划分 =====
    if ndim == 3:
        # 输入: B=批, Pl=气压层, Lat=纬, Lon=经, C=通道
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        # 第一步 view: 把每个空间维拆成 (窗口个数, 窗口内尺寸)
        # -> [B, nPl, win_pl, nLat, win_lat, nLon, win_lon, C],其中 nPl=Pl//win_pl 等
        x = x.view(
            B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C
        )
        # 第二步 permute+view:
        #   permute(0,5,1,3,2,4,6,7) 把"经度窗口个数 nLon"提到 batch 旁边,
        #   并把 (nPl,nLat) 三个窗口计数聚在一起、窗口内容 (win_pl,win_lat,win_lon,C) 放最后。
        #   最终 view 合并: 第一维 = B*nLon(经度当作独立 batch,因经度环状且偏置沿经平移不变),
        #                   第二维 = nPl*nLat(每个是一个待做注意力的"窗口位置"),后面是窗口内容。
        windows = (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        )
        return windows
    # ===== 二维窗口划分(同理,少了气压维) =====
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        # 拆成 (nLat, win_lat, nLon, win_lon)
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        # 把 nLon 提到 batch 旁,nLat 作为窗口位置维
        windows = (
            x.permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(-1, (Lat // win_lat), win_lat, win_lon, C)
        )
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    参数:
        windows: 输入窗口张量，形状为(B × num_lon, num_pl × num_lat, win_pl, win_lat, win_lon, C) 或(B × num_lon, num_lat, win_lat, win_lon, C)。
        window_size (tuple[int]): 窗口大小，格式为 [win_pl, win_lat, win_lon] 或 [win_lat, win_lon]。
        Pl (int): 气压层数（pressure levels）。
        Lat (int): 纬度（latitude）。
        Lon (int): 经度（longitude）。
        ndim (int): 窗口维度，取值为 3 或 2。
    返回值:
        x: 重建后的张量，形状为 (B, Pl, Lat, Lon, C) 或 (B, Lat, Lon, C)。
    """
    # ===== 三维还原(window_partition 的逆) =====
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        # 从第一维反推原始 batch: 第一维 = B*nLon,除掉 nLon=Lon//win_lon 就得到 B
        B = int(windows.shape[0] / (Lon / win_lon))
        # 先把打平的维拆回: [B, nLon, nPl, nLat, win_pl, win_lat, win_lon, C]
        x = windows.view(
            B,
            Lon // win_lon,
            Pl // win_pl,
            Lat // win_lat,
            win_pl,
            win_lat,
            win_lon,
            -1,
        )
        # permute 把 (窗口计数, 窗口内尺寸) 在每个空间维上交错排回原顺序,再 view 合并:
        #   (nPl,win_pl)->Pl, (nLat,win_lat)->Lat, (nLon,win_lon)->Lon
        # 得到 [B, Pl, Lat, Lon, C]
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
        return x
    # ===== 二维还原 =====
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Lat, Lon, -1)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    沿着经度（longitude）维度，最左和最右的索引实际上是相邻的。
    如果在最左和最右位置都出现了半个窗口（half window），则它们会被直接合并为一个完整窗口。
    参数:
        input_resolution (tuple[int]): 输入的分辨率，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        window_size (tuple[int]): 窗口大小，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        shift_size (tuple[int]): 用于SW-MSA的窗口平移大小，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        ndim (int): 窗口的维度，取值为 3 或 2

    返回:
        attn_mask: 注意力掩码张量，形状为(n_lon, n_pl * n_lat, win_pl * win_lat * win_lon, win_pl * win_lat * win_lon)或(n_lon, n_lat, win_lat * win_lon, win_lat * win_lon)。
    """
    # 先造一张"区域编号图" img_mask: 移窗后不同区块打上不同整数标签,同标签才允许互相注意
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size

        # 注意经度多开 shift_lon 的宽度: 为了处理经度环状"最左最右半窗合并"
        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    # 把每个维度切成三段: [整块区, 移窗过渡区, 边缘半窗区],三段的交叉组合就是不同"区域"
    if ndim == 3:
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    # 给每个区域组合填上递增编号 cnt
    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        # 裁掉多开的 shift_lon 宽度(把环状多出的部分切回真实经度范围)
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    # 用同一套窗口划分,把区域编号图切成一个个窗口
    mask_windows = window_partition(
        img_mask, window_size, ndim=ndim
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1 or n_lon, n_lat, win_lat, win_lon, 1
    # 窗口内格点数(展平成一行)
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    # 把每个窗口内的区域编号摊平成向量: [n_lon, n_pos, win_total]
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    # 两两相减: 同区域(编号相同)得 0,不同区域得非 0 -> 得到 [.., win_total, win_total] 的差矩阵
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    # 不同区域的对填 -100(softmax 后≈0,即禁止注意),同区域填 0(允许注意)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask
