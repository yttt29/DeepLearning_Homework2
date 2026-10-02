"""读取已有逐轮记录，绘制可用于报告的曲线；不训练、不读取测试集。"""
import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dlcv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = '#2563A6', '#D07A21'
SEEDS = (42, 43, 44)


def read_history(run, model, seed):
    """每行是一轮；转为数值数组，便于按列绘图。"""
    with (run / f'{model}_seed{seed}' / 'history.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert [int(r['epoch']) for r in rows] == list(range(1, 31))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def style():
    # macOS 的中文字体；其他系统可使用已安装的 CJK 字体。
    candidates = ('Arial Unicode MS', 'Noto Sans CJK SC', 'SimHei', 'Heiti TC')
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': next((x for x in candidates if x in available), 'DejaVu Sans'),
                         'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#333333', 'text.color': '#222222',
                         'svg.fonttype': 'path', 'savefig.facecolor': 'white'})


def finish(fig, axes, out, name, title, subtitle, legend, note):
    fig.suptitle(title, fontsize=21, y=0.985)
    fig.text(0.5, 0.936, subtitle, ha='center', fontsize=11)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.921),
               ncol=2, frameon=False)
    for ax in axes.flat:
        ax.grid(axis='y', color='#E5E5E5', linewidth=0.7)
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    for ax in axes[1]:
        ax.set_xlabel('训练轮次')
    fig.text(0.07, 0.022, note, fontsize=10, color='#555555')
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.115, top=0.80,
                        hspace=0.3, wspace=0.2)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(out / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def baseline_plot(run, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey='row')
    for col, (model, label) in enumerate((('softmax', 'Softmax'), ('linear2', '两层线性（无激活）'), ('mlp', 'ReLU MLP'))):
        histories = [read_history(run, model, s) for s in SEEDS]
        for row, metric in enumerate(('loss', 'accuracy')):
            ax = axes[row, col]
            for prefix, color, line in (('train_eval', BLUE, '-'), ('val', ORANGE, '--')):
                values = np.stack([h[f'{prefix}_{metric}'] for h in histories])
                if metric == 'accuracy':
                    values *= 100
                # 保留每个种子的细线，避免均值掩盖波动；粗线只是均值，不是置信区间。
                for values_one in values:
                    ax.plot(histories[0]['epoch'], values_one, color=color, linestyle=line,
                            linewidth=0.8, alpha=0.3)
                ax.plot(histories[0]['epoch'], values.mean(axis=0), color=color,
                        linestyle=line, linewidth=2)
            if row == 0:
                ax.set_title(label, pad=12)
    axes[0, 0].set_ylabel('平均交叉熵损失')
    axes[1, 0].set_ylabel('准确率（%）')
    legend = [Line2D([0], [0], color=BLUE, lw=2, label='训练集（轮末固定模型）'),
              Line2D([0], [0], color=ORANGE, lw=2, ls='--', label='验证集')]
    finish(fig, axes, out, 'baseline_learning_curves', '三模型的训练与验证曲线',
           'FashionMNIST · 训练 50,000 / 验证 10,000 · 种子 42、43、44 · 学习率 0.01',
           legend, '粗线：3 次运行的均值；细线：单次运行。所有点均用该轮结束时的模型评价；纵轴为局部范围。')


def diagnosis_plot(baseline, diagnosis, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey='row')
    for col, seed in enumerate(SEEDS):
        for run, color, line in ((baseline, BLUE, '-'), (diagnosis, ORANGE, '--')):
            h = read_history(run, 'linear2', seed)
            for row, key in enumerate(('train_eval_loss', 'val_loss')):
                axes[row, col].plot(h['epoch'], h[key], color=color, ls=line, lw=2)
        axes[0, col].set_title(f'种子 {seed}', pad=12)
        for row in range(2):
            axes[row, col].axvspan(21, 30, color='#EEEEEE', zorder=0)
    axes[0, 0].set_ylabel('静态训练损失')
    axes[1, 0].set_ylabel('验证损失')
    legend = [Line2D([0], [0], color=BLUE, lw=2, label='学习率 0.01（基线）'),
              Line2D([0], [0], color=ORANGE, lw=2, ls='--', label='学习率 0.003（对照）')]
    finish(fig, axes, out, 'diagnosis_learning_curves', '两层线性模型：学习率单因素对照',
           'FashionMNIST · 训练 50,000 / 验证 10,000 · 每次 30 轮 · 相同种子配对比较',
           legend, '灰色区间：预先指定的第 21–30 轮波动评价窗口。每列展示一个种子；纵轴为局部范围。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results/training/baseline_cpu_20261002')
    parser.add_argument('--diagnosis', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/figures')
    args = parser.parse_args()
    style()
    baseline_plot(args.baseline, args.out)
    if args.diagnosis:
        diagnosis_plot(args.baseline, args.diagnosis, args.out)
    print(f'曲线已保存：{args.out}')


if __name__ == '__main__':
    main()
