"""三模型共用的训练入口。先运行 python train.py --smoke 检查流程。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import ROOT, create_data_loaders
from models import MODEL_NAMES, build_model


def set_seed(seed):
    """固定训练随机数；数据加载器另外使用独立的同种子生成器。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(name):
    """显式选择设备；不可用时报错，不悄悄换设备影响比较。"""
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS 不可用，请使用 --device cpu')
    if name not in ('cpu', 'mps'):
        raise ValueError('本实验当前支持 cpu 或 mps')
    return torch.device(name)


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    """传入 optimizer 表示训练；不传则只做验证，不更新参数。"""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    # 验证时关闭梯度记录，节省内存；训练时开启自动求导。
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)  # 清掉上一批次的梯度。
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError('损失出现 NaN 或无穷大，停止并检查配置')
            if training:
                loss.backward()  # PyTorch 自动计算所有参数的梯度。
                optimizer.step()  # 用 SGD 更新权重和偏置。
            batch_size = labels.size(0)
            # 损失先乘样本数再汇总，避免最后一个小批次被赋予过高权重。
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size
    if count == 0:
        raise ValueError('没有读取到样本')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')


def train_one(name, seed, config, device, output_dir, smoke=False):
    """一个模型、一个种子的完整运行；每次重新创建数据加载器。"""
    set_seed(seed)
    train_loader, val_loader = create_data_loaders(config['batch_size'], seed)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=config['batch_size'], shuffle=False)
    model = build_model(name, config['hidden_dim'], seed).to(device)
    loss_fn = nn.CrossEntropyLoss()  # 输入 logits，不要先调用 Softmax。
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    run_dir = output_dir / f'{name}_seed{seed}'
    run_dir.mkdir()  # 已存在时不覆盖之前的训练结果。
    save_json(run_dir / 'config.json', {**config, 'model': name, 'seed': seed, 'smoke': smoke})
    initial_hash = hashlib.sha256()
    for parameter in model.parameters():
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    best_loss, best_epoch, best_accuracy = math.inf, 0, 0.0
    fields = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
              'train_samples', 'val_samples', 'epoch_seconds', 'train_eval_loss', 'train_eval_accuracy', 'train_eval_samples']
    started = time.perf_counter()
    with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.perf_counter()
            train = run_epoch(model, train_loader, loss_fn, device, optimizer,
                              max_batches=8 if smoke else None)
            val = run_epoch(model, val_loader, loss_fn, device,
                            max_batches=4 if smoke else None)
            train_eval = run_epoch(model, train_eval_loader, loss_fn, device,
                                   max_batches=8 if smoke else None)
            if device.type == 'mps':
                torch.mps.synchronize()  # 等 GPU 完成，保证计时覆盖实际计算。
            row = {'epoch': epoch, 'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
                   'train_samples': train['samples'], 'val_samples': val['samples'],
                   'epoch_seconds': time.perf_counter() - epoch_start,
                   'train_eval_loss': train_eval['loss'], 'train_eval_accuracy': train_eval['accuracy'],
                   'train_eval_samples': train_eval['samples']}
            writer.writerow(row)
            f.flush()  # 每轮落盘，中断后也能查看已完成的曲线数据。
            if val['loss'] < best_loss:  # 严格小于：并列时保留较早的轮次。
                best_loss, best_epoch, best_accuracy = val['loss'], epoch, val['accuracy']
                # 存为 CPU 张量，之后加载时不依赖保存模型所用的设备。
                torch.save({'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'model': name, 'seed': seed, 'hidden_dim': config['hidden_dim'],
                            'epoch': epoch, 'val_loss': best_loss, 'val_accuracy': best_accuracy},
                           run_dir / 'best.pt')
            print(f'{name} seed={seed} epoch={epoch}/{config["epochs"]} | '
                  f'train loss={train["loss"]:.4f} acc={train["accuracy"]:.2%} | '
                  f'val loss={val["loss"]:.4f} acc={val["accuracy"]:.2%}', flush=True)
    torch.save({'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'model': name, 'seed': seed, 'hidden_dim': config['hidden_dim'], 'epoch': config['epochs']},
               run_dir / 'last.pt')
    result = {'initial_parameters_sha256': initial_hash.hexdigest(), 'model': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
              'best_epoch': best_epoch, 'best_val_loss': best_loss, 'best_val_accuracy': best_accuracy,
              'training_and_validation_seconds': time.perf_counter() - started, 'smoke': smoke}
    save_json(run_dir / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps'])
    parser.add_argument('--output', type=Path, help='新输出目录；已存在则拒绝覆盖')
    parser.add_argument('--smoke', action='store_true', help='每模型最多 2 轮，每轮 8 批训练、4 批验证')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('models', 'seeds', 'epochs', 'device'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.smoke:
        config['epochs'] = min(config['epochs'], 2)
        if args.seeds is None:
            config['seeds'] = [42]
    for key in ('hidden_dim', 'batch_size', 'epochs', 'num_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} 必须为正整数')
    if (not config['models'] or any(m not in MODEL_NAMES for m in config['models'])
            or len(set(config['models'])) != len(config['models'])):
        raise ValueError('模型列表为空、有重复或包含未知模型')
    if (not config['seeds'] or any(type(s) is not int or not 0 <= s < 2**32 for s in config['seeds'])
            or len(set(config['seeds'])) != len(config['seeds'])):
        raise ValueError('种子必须是不重复的非负整数，且小于 2**32')
    if not all(math.isfinite(config[k]) for k in ('learning_rate', 'momentum', 'weight_decay')):
        raise ValueError('优化器配置必须是有限数值')
    if config['learning_rate'] <= 0 or not 0 <= config['momentum'] < 1 or config['weight_decay'] < 0:
        raise ValueError('学习率须为正，动量须在 [0, 1)，权重衰减须非负')
    device = select_device(config['device'])
    torch.set_num_threads(config['num_threads'])
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    output = args.output or ROOT / 'results' / ('smoke' if args.smoke else 'training') / stamp
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    source_hashes = {}
    for filename in ('train.py', 'models.py', 'data.py', 'evaluate.py', 'environment.yml', 'requirements-macos-arm64.lock.txt'):
        shutil.copy2(ROOT / filename, snapshot / filename)
        source_hashes[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    manifest = {'source_sha256': source_hashes, 'command': sys.argv, 'status': 'running', 'config': config, 'smoke': args.smoke,
                'torch_version': str(torch.__version__), 'python': platform.python_version(),
                'platform': platform.platform(), 'device': str(device),
                'split_sha256': hashlib.sha256((ROOT / 'data/split_seed42.npz').read_bytes()).hexdigest(),
                'selection_rule': 'minimum validation loss; earliest epoch on tie',
                'train_metric_note': 'train_* 为更新过程指标；train_eval_* 和 val_* 均用本轮结束模型计算，诊断泛化差距使用后两者。',
                'test_set_used': False}
    save_json(output / 'manifest.json', manifest)
    results = []
    try:
        for seed in config['seeds']:
            for name in config['models']:
                results.append(train_one(name, seed, config, device, output, args.smoke))
                save_json(output / 'summary.json', results)
    except BaseException as exc:
        manifest.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'manifest.json', manifest)
        raise
    manifest['status'] = 'completed'
    save_json(output / 'manifest.json', manifest)
    print(f'结果保存到：{output}')


if __name__ == '__main__':
    main()
