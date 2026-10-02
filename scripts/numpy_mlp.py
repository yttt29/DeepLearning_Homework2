"""仅用 NumPy 实现 MLP，并用中心差分检查手写梯度。

在实验目录运行：python scripts/numpy_mlp.py
本脚本只验证计算，不训练完整数据集，也不使用 PyTorch 自动求导。
"""
from pathlib import Path
import csv
import json
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_fixed_batch(batch_size=4):
    """读取已保存划分中的前几张训练图片，不接触验证集和测试集。"""
    raw = ROOT / 'data/FashionMNIST/raw'
    with (raw / 'train-images-idx3-ubyte').open('rb') as f:
        magic, count, rows, cols = struct.unpack('>IIII', f.read(16))
        if (magic, count, rows, cols) != (2051, 60000, 28, 28):
            raise ValueError('训练图像文件头不符合 Fashion-MNIST 格式')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows * cols)
    with (raw / 'train-labels-idx1-ubyte').open('rb') as f:
        magic, count = struct.unpack('>II', f.read(8))
        if (magic, count) != (2049, 60000):
            raise ValueError('训练标签文件头不符合 Fashion-MNIST 格式')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    with np.load(ROOT / 'data/split_seed42.npz') as split:
        indices = split['train'][:batch_size].copy()
    # 双精度可减少中心差分的舍入误差；像素缩放到 [0, 1]。
    return images[indices].astype(np.float64) / 255.0, labels[indices].astype(np.int64), indices


def initialize_parameters(input_dim=784, hidden_dim=8, classes=10, seed=42):
    """按输入宽度缩放随机权重，偏置从零开始。"""
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.normal(0, np.sqrt(1.0 / input_dim), (input_dim, hidden_dim)),
        'b1': np.zeros(hidden_dim, dtype=np.float64),
        'W2': rng.normal(0, np.sqrt(1.0 / hidden_dim), (hidden_dim, classes)),
        'b2': np.zeros(classes, dtype=np.float64),
    }


def forward(x, y, params):
    """线性层 → ReLU → 线性层 → Softmax 和平均交叉熵。"""
    z1 = x @ params['W1'] + params['b1']
    h = np.maximum(z1, 0)
    logits = h @ params['W2'] + params['b2']

    # 原始类别分数 logits，减去最大值，防止取指数时溢出浮点数表示范围。
    shifted = logits - logits.max(axis=1, keepdims=True)
    # 计算 e(logits)。
    exp_scores = np.exp(shifted)
    # 计算 e(logits) 之和。
    s = exp_scores.sum(axis=1, keepdims=True)
    # 计算 softmax 概率的对数。
    log_probs = shifted - np.log(s)
    # 计算 softmax 概率。
    probs = np.exp(log_probs)

    # 计算平均交叉熵损失，即预测正确的概率的负对数。
    loss = -log_probs[np.arange(len(y)), y].mean()
    # 将中间计算结果打包返回，为反向传播做准备。
    cache = {'x': x, 'y': y, 'z1': z1, 'h': h, 'probs': probs}
    return float(loss), cache


def backward(params, cache):
    """沿计算图反向应用链式法则，返回四组参数的梯度。"""
    x, y, z1, h, probs = (cache[k] for k in ('x', 'y', 'z1', 'h', 'probs'))
    
    # L 对 logits 的梯度为 (正确预测的概率 - one-hot 标签) / 批大小。
    d_logits = probs.copy()
    d_logits[np.arange(len(y)), y] -= 1
    d_logits /= len(y)

    # L 对第二个线性层权重的梯度。
    d_w2 = h.T @ d_logits
    # L 对第二个线性层偏置的梯度。
    d_b2 = d_logits.sum(axis=0)  # 同一偏置被批次内所有样本共享。
    # L 对 ReLU 的输出的梯度。
    d_h = d_logits @ params['W2'].T
    # L 对第一个线性层的输出的梯度。
    d_z1 = d_h * (z1 > 0)  # ReLU 在正半轴导数为 1，负半轴为 0。
    return {
        'W1': x.T @ d_z1,       # 第一个线性层的权重的梯度。
        'b1': d_z1.sum(axis=0), # 第一个线性层的偏置的梯度。
        'W2': d_w2,
        'b2': d_b2,
    }


def check_gradients(x, y, params, epsilon=1e-5, samples_per_tensor=20,
                    seed=43, atol=1e-7, rtol=1e-5):
    """用中心差分检查刚才计算出的梯度是否正确。"""
    if epsilon <= 0 or samples_per_tensor < 1:
        raise ValueError('epsilon 和 samples_per_tensor 必须为正数')
    loss, cache = forward(x, y, params)
    analytic = backward(params, cache)
    rng = np.random.default_rng(seed)
    records = []
    for name, values in params.items():
        chosen = rng.choice(values.size, min(samples_per_tensor, values.size), replace=False)
        for flat_index in chosen:
            index = np.unravel_index(int(flat_index), values.shape)
            original = values[index].copy()
            try:
                values[index] = original + epsilon
                loss_plus, plus = forward(x, y, params)
                values[index] = original - epsilon
                loss_minus, minus = forward(x, y, params)
            finally:
                values[index] = original

            # 中心差分跨过 ReLU 的折点时不能用来判定反向传播是否正确。
            affected = plus['z1'] != minus['z1']
            crosses_kink = np.any(affected & (np.minimum(plus['z1'], minus['z1']) <= 0)
                                  & (np.maximum(plus['z1'], minus['z1']) >= 0))
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            actual = float(analytic[name][index])
            absolute = abs(actual - numerical)
            relative = absolute / max(1e-12, abs(actual) + abs(numerical))
            finite = np.isfinite([actual, numerical, absolute]).all()
            # 梯度接近零时采用绝对容差，避免相对误差被放大。
            passed = finite and absolute <= atol + rtol * max(abs(actual), abs(numerical))
            records.append({
                'parameter': name, 'index': ','.join(map(str, index)),
                'analytic': actual, 'numerical': numerical,
                'absolute_error': absolute, 'relative_error': relative,
                'status': 'skipped_relu_kink' if crosses_kink else ('passed' if passed else 'failed'),
            })
    summary = {}
    for name in params:
        checked = [r for r in records if r['parameter'] == name and r['status'] != 'skipped_relu_kink']
        summary[name] = {
            'checked': len(checked),
            'skipped': sum(r['parameter'] == name and r['status'] == 'skipped_relu_kink' for r in records),
            'failed': sum(r['status'] == 'failed' for r in checked),
            'max_absolute_error': max((r['absolute_error'] for r in checked), default=None),
            'max_relative_error': max((r['relative_error'] for r in checked), default=None),
        }
    passed = all(s['checked'] > 0 and s['failed'] == 0 for s in summary.values())
    return {'loss': loss, 'passed': passed, 'epsilon': epsilon, 'atol': atol,
            'rtol': rtol, 'sampling_seed': seed, 'summary': summary, 'records': records}


def main():
    x, y, indices = load_fixed_batch()
    params = initialize_parameters()

    # 前向、反向、梯度检查
    report = check_gradients(x, y, params)
    report.update({'network': [784, 8, 10], 'activation': 'ReLU', 'dtype': 'float64',
                   'initialization_seed': 42, 'batch_indices': indices.tolist(), 'labels': y.tolist(),
                   'note': '抽样检查，仅覆盖所选参数与当前批次；不是全部输入上的正确性证明。'})

    # 保存检查结果
    out = ROOT / 'results/gradient_check'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    # 打印检查结果
    with (out / 'details.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['records'][0]))
        writer.writeheader()
        writer.writerows(report['records'])
    print(f"固定训练样本索引：{indices.tolist()}；平均交叉熵：{report['loss']:.8f}")
    for name, result in report['summary'].items():
        print(f"{name}: 检查 {result['checked']} 项，跳过 {result['skipped']} 项，"
              f"失败 {result['failed']} 项，最大绝对误差 {result['max_absolute_error']}")
    print('梯度检查：' + ('通过' if report['passed'] else '未通过'))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
