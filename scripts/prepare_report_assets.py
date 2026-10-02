"""核对最终测试预测，汇总报告数据并绘制混淆矩阵与错误样例。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import FashionMNIST

from plot_learning_curves import plt, style
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('softmax', 'linear2', 'mlp')
TITLES = ('Softmax', '两层线性', 'ReLU MLP')
CLASSES = ('短袖上衣', '长裤', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    analysis = ROOT / 'results/analysis'
    figures = ROOT / 'results/figures'
    plan = read_json(analysis / 'final_evaluation_plan.json')
    baseline, diagnosis = [ROOT / x for x in plan['runs']]
    test = FashionMNIST(ROOT / 'data', train=False, download=False)
    labels = test.targets.numpy()
    assert np.array_equal(np.bincount(labels), np.full(10, 1000))
    records, aggregates, predictions, confusions = [], {}, {}, {}
    for run in (baseline, diagnosis):
        manifest = read_json(run / 'manifest.json')
        condition = 'baseline' if run == baseline else 'diagnosis'
        for model in manifest['config']['models']:
            recalls = []
            group = []
            for seed in manifest['config']['seeds']:
                directory = run / f'{model}_seed{seed}'
                checkpoint = directory / 'best.pt'
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan['checkpoints'][str(checkpoint.relative_to(ROOT))]
                m = read_json(directory / 'test_evaluation/metrics.json')
                val = read_json(directory / 'validation_evaluation/metrics.json')
                assert m['checkpoint_sha256'] == val['checkpoint_sha256']
                assert m['selected_epoch'] == val['selected_epoch']
                with np.load(directory / 'test_evaluation/predictions.npz') as saved:
                    p = {k: saved[k].copy() for k in saved.files}
                assert np.array_equal(p['indices'], np.arange(10000))
                assert np.array_equal(p['labels'], labels)
                assert np.array_equal(p['predictions'], p['logits'].argmax(axis=1))
                assert np.isfinite(p['logits']).all()
                np.testing.assert_allclose(p['probabilities'].sum(axis=1), 1, atol=1e-6)
                # 从原始 logits 独立重算交叉熵，避免只信导出汇总。
                shifted = p['logits'].astype(np.float64) - p['logits'].max(axis=1, keepdims=True)
                log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                ce = -log_probs[np.arange(10000), labels].mean()
                assert abs(ce - m['loss']) < 1e-6
                matrix = np.zeros((10, 10), dtype=int)
                np.add.at(matrix, (labels, p['predictions']), 1)
                with (directory / 'test_evaluation/confusion_matrix.csv').open() as f:
                    rows = list(csv.reader(f))
                assert np.array_equal(matrix, np.array([r[1:] for r in rows[1:]], dtype=int))
                accuracy = float(np.trace(matrix) / 10000)
                recall = matrix.diagonal() / matrix.sum(axis=1)
                assert accuracy == m['accuracy'] and abs(recall.mean() - m['macro_recall']) < 1e-12
                with (directory / 'test_evaluation/per_class.csv').open() as f:
                    per_class = list(csv.DictReader(f))
                np.testing.assert_allclose(recall, [float(x['recall']) for x in per_class], atol=1e-12)
                record = {'condition': condition, 'model': model, 'seed': seed,
                          'epoch': m['selected_epoch'], 'validation_accuracy': val['accuracy'],
                          'test_accuracy': accuracy, 'test_loss': float(ce), 'test_errors': m['errors']}
                records.append(record)
                group.append(record)
                recalls.append(recall)
                if condition == 'baseline' and seed == plan['representative_seed']:
                    predictions[model], confusions[model] = p, matrix
            aggregates[f'{condition}_{model}'] = {
                k: {'mean': float(np.mean([r[k] for r in group])),
                    'sample_std': float(np.std([r[k] for r in group], ddof=1))}
                for k in ('validation_accuracy', 'test_accuracy', 'test_loss')}
            aggregates[f'{condition}_{model}']['recall_mean'] = np.mean(recalls, axis=0).tolist()
            aggregates[f'{condition}_{model}']['recall_sample_std'] = np.std(recalls, axis=0, ddof=1).tolist()

    style()
    # 相同色标、按行归一化；每个对角格就是该类别召回率。
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    cmap = LinearSegmentedColormap.from_list('report_blue', ['#FFFFFF', '#2563A6'])
    for ax, model, title in zip(axes, NAMES, TITLES):
        matrix = confusions[model] / 10  # 每个真实类别恰有1000张，换成百分比。
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f'{matrix[i,j]:.1f}' if matrix[i,j] else '0',
                        ha='center', va='center', fontsize=7.7,
                        color='white' if matrix[i,j] >= 65 else '#333333')
        ax.set_xticks(range(10), CLASSES, rotation=65, ha='right', fontsize=9)
        ax.set_yticks(range(10), CLASSES, fontsize=9)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
    fig.suptitle('三模型测试集混淆矩阵', fontsize=21, y=.98)
    fig.text(.5,.91,'固定种子 42 · 每模型 10,000 张测试图像 · 每行归一化为百分比',ha='center',fontsize=11)
    fig.subplots_adjust(left=.065,right=.935,bottom=.21,top=.83,wspace=.31)
    cax=fig.add_axes([.95,.27,.012,.5])
    fig.colorbar(im,cax=cax,label='占真实类别的比例（%）')
    for ext in ('png','svg'):
        fig.savefig(figures/f'test_confusion_matrices.{ext}',dpi=190)
    plt.close(fig)

    # 按预先保存的规则选择样例，不按视觉效果或模型表现挑选。
    cm = confusions['softmax']
    pairs = sorted(((int(cm[i,j]),i,j) for i in range(10) for j in range(10) if i != j and cm[i,j]),
                   key=lambda x:(-x[0],x[1],x[2]))[:6]
    examples=[]
    fig, axes = plt.subplots(2,3,figsize=(12,10))
    for ax,(count,truth,predicted) in zip(axes.flat,pairs):
        idx=int(np.flatnonzero((labels==truth)&(predictions['softmax']['predictions']==predicted))[0])
        ax.imshow(test.data[idx].numpy(),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'样本 {idx} · 真实：{CLASSES[truth]}',fontsize=12,pad=8)
        details=[]; sample={'index':idx,'true_label':truth,'pair_count':count,'predictions':{}}
        for model,title in zip(NAMES,TITLES):
            predicted_class=int(predictions[model]['predictions'][idx])
            confidence=float(predictions[model]['probabilities'][idx,predicted_class])
            details.append(f'{title}：{CLASSES[predicted_class]}（{confidence:.1%}）')
            sample['predictions'][model]={'class':predicted_class,'confidence':confidence}
        ax.set_xlabel('\n'.join(details),fontsize=11,labelpad=9)
        examples.append(sample)
    fig.suptitle('常见混淆的测试图像及配对预测',fontsize=21,y=.98)
    fig.text(.5,.935,'固定种子 42 · Softmax 前 6 个有向混淆类别对 · 每对取原始索引最小的错误样本',ha='center',fontsize=10)
    fig.subplots_adjust(left=.07,right=.95,top=.86,bottom=.12,hspace=.80,wspace=.35)
    fig.text(.08,.025,'括号内为模型对预测类别给出的概率；这不是预测必然正确的保证。',fontsize=10)
    fig.savefig(figures/'test_error_examples.png',dpi=180)
    plt.close(fig)

    original = FashionMNIST(ROOT / 'data', train=True, download=False)
    with np.load(ROOT / 'data/split_seed42.npz') as split:
        counts={k:np.bincount(original.targets.numpy()[split[k]],minlength=10).tolist() for k in ('train','validation')}
    result={'records':records,'aggregates':aggregates,'classes':list(CLASSES),'split_class_counts':counts,
            'representative_seed':plan['representative_seed'], 'error_examples':examples,
            'top_softmax_confusions':[{'count':n,'true_class':i,'predicted_class':j} for n,i,j in pairs],
            'audit':{'passed':True,'checked_models':len(records),'samples_per_model':10000,
                     'fixed_checkpoints_match_plan':True,'no_post_test_training':True,
                     'predictions_metrics_and_confusion_agree':True,'test_labels_match_official':True}}
    write_json(analysis/'report_data.json',result)
    with (analysis/'final_performance.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({'aggregates':aggregates,'error_examples':examples,'audit':result['audit']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
