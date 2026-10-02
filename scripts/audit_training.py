"""核对完整训练的轮数、样本数、最佳模型、验证导出与受控初始化。"""
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and not manifest['smoke']
    config = manifest['config']
    summaries = json.loads((args.run / 'summary.json').read_text())
    expected = {(m, s) for m in config['models'] for s in config['seeds']}
    assert {(r['model'], r['seed']) for r in summaries} == expected
    assert len(summaries) == len(expected)
    for summary in summaries:
        run = args.run / f"{summary['model']}_seed{summary['seed']}"
        with (run / 'history.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == config['epochs']
        assert [int(r['epoch']) for r in rows] == list(range(1, config['epochs']+1))
        for row in rows:
            assert int(row['train_samples']) == int(row['train_eval_samples']) == 50000
            assert int(row['val_samples']) == 10000
            assert all(math.isfinite(float(v)) for v in row.values())
            assert all(0 <= float(row[k]) <= 1 for k in ('train_accuracy', 'train_eval_accuracy', 'val_accuracy'))
        selected = min(rows, key=lambda r: float(r['val_loss']))
        best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        assert last['epoch'] == config['epochs']
        assert best['epoch'] == summary['best_epoch'] == int(selected['epoch'])
        assert abs(best['val_loss']-float(selected['val_loss'])) < 1e-8
        evaluation = run / 'validation_evaluation'
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        assert abs(metrics['accuracy']-summary['best_val_accuracy']) < 1e-12
        assert abs(metrics['loss']-summary['best_val_loss']) < 1e-5
        with np.load(evaluation / 'predictions.npz') as p:
            assert p['probabilities'].shape == p['logits'].shape == (10000, 10)
            assert len(np.unique(p['indices'])) == 10000
            np.testing.assert_allclose(p['probabilities'].sum(axis=1), 1, atol=1e-6)
            assert np.array_equal(p['predictions'], p['logits'].argmax(axis=1))
        with (evaluation / 'per_class.csv').open() as f:
            classes = list(csv.DictReader(f))
        assert len(classes) == 10 and sum(int(r['support']) for r in classes) == 10000
    # 单模型诊断没有另一种模型；只有两者均运行时才检查配对初始化。
    paired_checked = {'linear2', 'mlp'}.issubset(config['models'])
    if paired_checked:
        for seed in config['seeds']:
            paired = [r['initial_parameters_sha256'] for r in summaries if r['seed'] == seed and r['model'] in ('linear2', 'mlp')]
            assert len(paired) == 2 and paired[0] == paired[1]
    report = {'passed': True, 'completed_runs': len(summaries), 'total_epochs': len(summaries)*config['epochs'],
              'checks': ['完整样本数与轮次', '指标有限且准确率范围正确', '验证损失最低轮次选择',
                         '最佳与最后模型文件', '重新加载预测与训练日志一致', '逐类支持数与逐样本概率'],
              'paired_initialization_checked': paired_checked, 'test_set_used': False}
    if paired_checked:
        report['checks'].append('两个两层模型的初始化完全相同')
    (args.run / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
