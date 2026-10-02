"""加载 Fashion-MNIST，为训练和验证提供一批一批的数据。

在实验目录运行：python data.py
这里只读取训练集及其固定划分，不加载测试集，也不训练模型。
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

# 根据本文件的位置寻找数据，不依赖终端当前所在目录。
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'


def load_datasets():
    """读取 60,000 张原始训练图片，按已有索引分成训练集和验证集。"""
    # 数据已经下载好了；download=False 避免运行时再次联网。
    original = FashionMNIST(root=DATA_DIR, train=True, download=False)

    # 每张 28×28 图片展平为 784 维，并把整数像素转为 [0, 1] 的小数。
    images = original.data.reshape(len(original), 784).to(torch.float32) / 255.0
    labels = original.targets.to(torch.int64)
    dataset = TensorDataset(images, labels)  # 每个样本是一对 (图片, 标签)。

    # 复用之前保存的划分，不能每次训练时重新随机划分。
    with np.load(DATA_DIR / 'split_seed42.npz') as split:
        train_indices = split['train'].copy()
        val_indices = split['validation'].copy()

    # 确保两个集合无重复、无遗漏，也没有越界的索引。
    combined = np.concatenate([train_indices, val_indices])
    if (train_indices.shape != (50000,) or val_indices.shape != (10000,)
            or not np.issubdtype(combined.dtype, np.integer)
            or not np.array_equal(np.sort(combined), np.arange(len(dataset)))):
        raise ValueError('划分索引必须无重复地覆盖全部 60,000 个样本，数量为 50,000 / 10,000')

    # Subset 只保存索引，两个子集共用原始数据，不复制整套图片。
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    return train_dataset, val_dataset


def create_data_loaders(batch_size=128, seed=42):
    """返回训练和验证加载器；seed 只控制训练顺序，不改变数据划分。"""
    if batch_size < 1:
        raise ValueError('batch_size 必须为正整数')
    train_dataset, val_dataset = load_datasets()

    # 独立随机数生成器：模型初始化消耗随机数时，不会影响样本顺序。
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 每轮遍历都会重新打乱，但固定种子可复现整个顺序。
        generator=generator,
        num_workers=0,      # 先在主进程加载，便于理解和调试。
        drop_last=False,    # 最后一批不足 128 个也保留，不遗漏样本。
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # 验证时保持固定顺序。
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def main():
    """只取一个批次查看数据，不进行训练。"""
    train_loader, val_loader = create_data_loaders()
    images, labels = next(iter(train_loader))
    print(f'训练样本：{len(train_loader.dataset)}；验证样本：{len(val_loader.dataset)}')
    print(f'训练批次数：{len(train_loader)}；验证批次数：{len(val_loader)}')
    print(f'一批图片：形状 {tuple(images.shape)}，类型 {images.dtype}')
    print(f'一批标签：形状 {tuple(labels.shape)}，类型 {labels.dtype}')
    print(f'本批像素范围：[{images.min().item():.1f}, {images.max().item():.1f}]')
    print(f'前 8 个标签：{labels[:8].tolist()}')


if __name__ == '__main__':
    main()
