"""三个分类模型；forward 返回原始分数，由交叉熵处理 Softmax。"""
import torch
from torch import nn

MODEL_NAMES = ('softmax', 'linear2', 'mlp')


class SoftmaxClassifier(nn.Module):
    """线性基线：784 → 10。"""
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        return self.output(x)


class TwoLayerClassifier(nn.Module):
    """两个线性层；use_relu 决定中间是否使用 ReLU。"""
    def __init__(self, hidden_dim=256, use_relu=True):
        super().__init__()
        self.hidden = nn.Linear(784, hidden_dim)
        self.activation = nn.ReLU() if use_relu else nn.Identity()
        self.output = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        h = self.hidden(x)
        h = self.activation(h)  # Identity 原样返回输入，不引入非线性。
        return self.output(h)


def build_model(name, hidden_dim=256, seed=42):
    """创建并初始化模型；同一种子下，两个两层模型初始参数逐项相同。"""
    if name not in MODEL_NAMES:
        raise ValueError(f'未知模型：{name}；请选择 {MODEL_NAMES}')
    if hidden_dim < 1:
        raise ValueError('hidden_dim 必须为正整数')
    model = SoftmaxClassifier() if name == 'softmax' else TwoLayerClassifier(hidden_dim, name == 'mlp')
    generator = torch.Generator().manual_seed(seed)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # 三模型统一使用 Xavier 均匀初始化、零偏置。
            # 两层模型使用相同形状和随机序列，只改变是否经过 ReLU。
            nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=generator)
            nn.init.zeros_(layer.bias)
    return model
