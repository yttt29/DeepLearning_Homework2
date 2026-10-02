"""同一代码和数据比较 CPU/MPS；每模型设备各一轮预热、三轮完整计时。"""
from pathlib import Path
import sys
import csv
import json
import statistics
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import ROOT
from train import train_one, select_device, save_json


def main():
    config = json.loads((ROOT / 'configs/baseline.json').read_text())
    config.update(epochs=4, seeds=[42])
    torch.set_num_threads(config['num_threads'])
    out = ROOT / 'results/device_benchmark'
    out.mkdir(exist_ok=False)
    rows = []
    for i, name in enumerate(config['models']):
        # 交替设备顺序，减小总是先运行同一设备的影响。
        for dev in (['cpu', 'mps'] if i % 2 == 0 else ['mps', 'cpu']):
            device_dir = out / dev
            device_dir.mkdir(exist_ok=True)
            current = {**config, 'device': dev}
            train_one(name, 42, current, select_device(dev), device_dir)
            with (device_dir / f'{name}_seed42/history.csv').open() as f:
                history = list(csv.DictReader(f))
            samples = [float(r['epoch_seconds']) for r in history[1:]]
            rows.append({'model': name, 'device': dev, 'warmup_seconds': float(history[0]['epoch_seconds']),
                         'measured_epoch_seconds': samples, 'median_epoch_seconds': statistics.median(samples)})
            save_json(out / 'measurements.json', rows)
    totals = {d: sum(r['median_epoch_seconds'] for r in rows if r['device'] == d) for d in ('cpu', 'mps')}
    selected = min(totals, key=totals.get)
    report = {'selected_device': selected, 'measurements': rows, 'sum_model_epoch_seconds': totals,
              'estimated_9_runs_30_epochs_seconds': totals[selected] * 30 * 3,
              'scope': '每轮 50k 训练 + 10k 验证 + 50k 训练集静态评价；4 CPU 线程；同初始化及数据顺序；不使用测试集。',
              'caveat': '比较当前代码端到端性能，不代表硬件峰值；中位数不含保存模型与启动耗时。'}
    save_json(out / 'report.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
