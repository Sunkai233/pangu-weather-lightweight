# =============================================================================
# 文件: pangu_layer.py
# 作用: 两个通用小组件 —— DropPath(随机深度)与 Mlp(Transformer 里的前馈网络)。
#       它们被主干的每个 Transformer 块反复调用,是搭积木用的基础层。
# 在网络中的位置: 底层零件,不直接接触气象数据的形状,作用在特征张量上。
# 关键点:
#   1) DropPath = 随机深度: 训练时按整条样本随机"跳过"残差分支(不是随机丢单个神经元),推理时原样通过。
#   2) Mlp = fc1 -> GELU 激活 -> dropout -> fc2 -> dropout;中间层维度默认等于输入(通常放大到 4 倍)。
#   3) 真正的 drop_path 计算逻辑在 func_utils.pangu_utils 里,这里的 DropPath 只是它的 nn.Module 包装。
# (本文件为官方原始代码逐行加注释版,代码零改动,仅新增中文注释)
# =============================================================================
import torch
from torch import nn
# 从工具模块导入底层的 drop_path 函数(真正做随机丢弃的逻辑)
from ..func_utils.pangu_utils import drop_path


class DropPath(nn.Module):
    """摘自 timm 仓库
    按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）当应用于残差块的主路径时）
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        # drop_prob: 丢弃概率; scale_by_keep: 是否按存活率放大以保持期望不变
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        # 直接委托给底层函数,把当前是否 training 传进去(推理时不丢)
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        # print(model) 时显示 drop_prob,便于调试
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


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
        # 隐藏层维度默认等于输入维度(实际用时常传入 in_features*mlp_ratio,如 4 倍)
        hidden_features = hidden_features or in_features
        # 第一层线性: 升维到 hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 非线性激活,默认 GELU
        self.act = act_layer()
        # 第二层线性: 降回 out_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        # dropout 层(fc1 后和 fc2 后各用一次)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        # 升维
        x = self.fc1(x)
        # 激活
        x = self.act(x)
        # dropout
        x = self.drop(x)
        # 降维
        x = self.fc2(x)
        # dropout
        x = self.drop(x)
        return x
