"""按训练前保存的计划，核对单因素对照并计算配对结果。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(run, model, seed):
    directory = run / f'{model}_seed{seed}'
    with (directory / 'history.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['epoch']) for r in rows] == list(range(1, 31))
    val = np.array([float(r['val_loss']) for r in rows])
    train = np.array([float(r['train_eval_loss']) for r in rows])
    selected = int(np.argmin(val))
    evaluation = read_json(directory / 'validation_evaluation/metrics.json')
    assert abs(evaluation['loss'] - val[selected]) < 1e-5
    assert evaluation['accuracy'] == float(rows[selected]['val_accuracy'])
    return {'late_val_step': float(np.abs(np.diff(val[-10:])).mean()),
            'late_train_step': float(np.abs(np.diff(train[-10:])).mean()),
            'late_val_range': float(np.ptp(val[-10:])),
            'best_val_loss': float(val[selected]),
            'best_val_accuracy': float(rows[selected]['val_accuracy']),
            'best_epoch': selected + 1,
            'final_val_loss': float(val[-1]),
            'final_val_accuracy': float(rows[-1]['val_accuracy']),
            'final_best_loss_gap': float(val[-1] - val[selected])}


def main():
    out = ROOT / 'results/analysis'
    plan = read_json(out / 'diagnosis_plan.json')
    baseline, diagnosis = [ROOT / plan[k] for k in ('baseline', 'diagnosis')]
    base_manifest, diag_manifest = [read_json(r / 'manifest.json') for r in (baseline, diagnosis)]
    assert all(m['status'] == 'completed' and not m['smoke'] and not m['test_set_used']
               for m in (base_manifest, diag_manifest))
    # 除了选择需要诊断的模型，唯一改变的训练参数必须是学习率。
    config_a = base_manifest['config'].copy()
    config_b = diag_manifest['config'].copy()
    assert config_b.pop('models') == [plan['model']]
    assert plan['model'] in config_a.pop('models')
    changed = [k for k in config_a if config_a[k] != config_b[k]]
    assert set(config_a) == set(config_b) and changed == [plan['factor']]
    assert config_a['learning_rate'] == plan['before'] and config_b['learning_rate'] == plan['after']
    assert config_b['seeds'] == plan['seeds'] and config_b['epochs'] == plan['epochs']
    for key in ('split_sha256', 'source_sha256', 'selection_rule', 'device', 'torch_version', 'python'):
        assert base_manifest[key] == diag_manifest[key], key
    for path, digest in plan['baseline_history_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest

    records, paired = [], []
    for seed in plan['seeds']:
        initials = [read_json(run / f"{plan['model']}_seed{seed}/summary.json")['initial_parameters_sha256']
                    for run in (baseline, diagnosis)]
        assert initials[0] == initials[1]
        a, b = [metrics(run, plan['model'], seed) for run in (baseline, diagnosis)]
        for rate, values in ((plan['before'], a), (plan['after'], b)):
            records.append({'seed': seed, 'learning_rate': rate, **values})
        paired.append({'seed': seed, **{f'{k}_delta': b[k] - a[k] for k in a},
                       'late_val_step_reduction_percent': 100 * (1 - b['late_val_step'] / a['late_val_step'])})

    aggregate = {}
    for rate in (plan['before'], plan['after']):
        group = [r for r in records if r['learning_rate'] == rate]
        aggregate[str(rate)] = {k: {'mean': float(np.mean([r[k] for r in group])),
                                   'sample_std': float(np.std([r[k] for r in group], ddof=1))}
                                for k in group[0] if k not in ('seed', 'learning_rate')}
    mean_a = aggregate[str(plan['before'])]['late_val_step']['mean']
    mean_b = aggregate[str(plan['after'])]['late_val_step']['mean']
    report = {'audit_passed': True, 'baseline': plan['baseline'], 'diagnosis': plan['diagnosis'],
              'plan_sha256': hashlib.sha256((out / 'diagnosis_plan.json').read_bytes()).hexdigest(),
              'changed_training_factors': changed, 'source_split_and_initialization_equal': True,
              'metric_definition': plan['primary_metric'], 'records': records, 'paired': paired,
              'aggregate': aggregate, 'reduction_in_mean_late_val_step_percent': 100 * (1 - mean_b / mean_a),
              'all_seeds_less_volatile': all(p['late_val_step_delta'] < 0 for p in paired),
              'test_set_used': False,
              'caveats': ['只比较3个种子和一个固定划分，不做总体显著性结论。',
                          '稳定性与最佳验证性能分别评价；降低学习率可能减慢早期收敛。',
                          '同一验证集用于选轮次和诊断；最终测试结果另行评价。']}
    (out / 'diagnosis_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for filename, rows in (('diagnosis_metrics.csv', records), ('diagnosis_paired.csv', paired)):
        with (out / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ('audit_passed', 'aggregate', 'reduction_in_mean_late_val_step_percent', 'all_seeds_less_volatile')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
