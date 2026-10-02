"""保存逐样本预测、混淆矩阵、逐类召回率和错误索引，供后续报告画图。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

from data import DATA_DIR, load_datasets
from models import build_model


def export_predictions(model, loader, indices, device, output_dir, split_name, classes):
    """按加载器固定顺序保存结果；调用者须提供与之对应的原始样本索引。"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    if len(indices) != len(labels):
        raise ValueError('样本索引与预测数量不同')
    probs = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    loss = nn.functional.cross_entropy(logits, labels).item()
    truth, pred = labels.numpy(), predictions.numpy()
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (truth, pred), 1)  # 行是真实类别，列是预测类别。
    support = confusion.sum(axis=1)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(10, dtype=float), where=support != 0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / 'predictions.npz', indices=np.asarray(indices),
                        labels=truth, predictions=pred, logits=logits.numpy(), probabilities=probs.numpy())
    with (output_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['true_class / predicted_class', *classes])
        writer.writerows([[classes[i], *row.tolist()] for i, row in enumerate(confusion)])
    with (output_dir / 'per_class.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['class_id', 'class_name', 'support', 'correct', 'recall'])
        writer.writerows([[i, classes[i], int(support[i]), int(confusion[i, i]), float(recall[i])] for i in range(10)])
    confidence = probs.max(dim=1).values.numpy()
    errors = np.flatnonzero(truth != pred)
    errors = errors[np.argsort(-confidence[errors], kind='stable')]
    with (output_dir / 'errors.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['original_index', 'true_label', 'predicted_label', 'confidence'])
        writer.writerows([[int(indices[i]), int(truth[i]), int(pred[i]), float(confidence[i])] for i in errors])
    metrics = {'split': split_name, 'samples': len(labels), 'loss': loss,
               'accuracy': float((truth == pred).mean()), 'macro_recall': float(recall.mean()),
               'errors': len(errors), 'classes': list(classes), 'confusion_orientation': 'rows=true, columns=predicted'}
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='含 manifest.json 的训练结果目录')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['smoke']:
        raise ValueError('只评价完整结束的正式训练，不使用短程检查结果')
    if args.split == 'test':
        original = FashionMNIST(DATA_DIR, train=False, download=False)
        dataset = TensorDataset(original.data.reshape(-1, 784).float() / 255, original.targets.long())
        indices = list(range(len(dataset)))
    else:
        _, dataset = load_datasets()
        indices = dataset.indices
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    results = []
    torch.set_num_threads(manifest['config']['num_threads'])
    for checkpoint_path in sorted(args.run.glob('*/best.pt')):
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = build_model(state['model'], state['hidden_dim'], state['seed'])
        model.load_state_dict(state['model_state'])
        output = checkpoint_path.parent / f'{args.split}_evaluation'
        metrics = export_predictions(model, loader, indices, torch.device('cpu'), output,
                                     args.split, FashionMNIST.classes)
        metrics.update(model=state['model'], seed=state['seed'], selected_epoch=state['epoch'],
                       checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        results.append(metrics)
        print(state['model'], state['seed'], args.split, metrics['accuracy'], flush=True)
    with (args.run / f'{args.split}_performance.csv').open('w', newline='', encoding='utf-8') as f:
        keys = ['model', 'seed', 'selected_epoch', 'samples', 'loss', 'accuracy', 'macro_recall']
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
    aggregates = []
    for name in manifest['config']['models']:
        rows = [r for r in results if r['model'] == name]
        values = [r['accuracy'] for r in rows]
        aggregates.append({'model': name, 'runs': len(rows), 'accuracy_mean': float(np.mean(values)),
                           'accuracy_sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    (args.run / f'{args.split}_aggregate.json').write_text(json.dumps(aggregates, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
