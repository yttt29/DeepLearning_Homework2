# 从线性分类器到多层感知机

《深度学习与计算机视觉》课程实验。徐义涛，U202314356，种子2301班。

在 Fashion-MNIST 上比较 Softmax 分类器、两层无激活线性网络与 ReLU MLP，完成手写反向传播和梯度检查，并通过学习率单因素实验分析训练波动。

## 提交内容与结果入口

本仓库包含可运行源代码、README 和最终结果图表。

| 课程要求 | 对应文件 |
|---|---|
| 可运行源代码 | 根目录四个 Python 文件、`scripts/`、`configs/`、`environment.yml` |
| 环境、命令、配置与随机种子说明 | 本 README |
| PDF 实验报告 | `实验报告.pdf`（待补充） |
| 梯度检查结果 | [汇总](results/gradient_check/report.json)、[逐项结果](results/gradient_check/details.csv) |
| 三模型性能表 | [性能汇总表](results/summary/performance.csv) |
| 训练与验证曲线 | [三模型曲线](results/figures/baseline_learning_curves.png) |
| 混淆矩阵 | [三模型测试集混淆矩阵](results/figures/test_confusion_matrices.png) |
| 逐类召回率 | [召回率表](results/summary/per_class_recall.csv) |
| 至少一组典型错误 | [六个典型错误样例](results/figures/test_error_examples.png) |
| 单因素训练诊断 | [学习率对照曲线](results/figures/diagnosis_learning_curves.png) |

仓库不附模型权重、逐样本预测、完整训练日志、源码快照、测速记录或原始数据集。这些过程文件不是课程指定的提交内容；重新运行代码时会在本地生成所需数据。

## 实验结果

下表为三个种子的均值 ± 样本标准差，准确率与标准差的单位均为百分点。

| 模型 | 学习率 | 测试准确率 |
|---|---:|---:|
| Softmax | 0.01 | 84.33% ± 0.11 |
| 两层线性，无激活 | 0.01 | 84.30% ± 0.20 |
| ReLU MLP | 0.01 | 88.15% ± 0.20 |
| 两层线性诊断 | 0.003 | 84.36% ± 0.14 |

CSV 表中的准确率、召回率及其标准差采用 0–1 比例；乘以 100 后得到百分数或百分点。

梯度检查共 58 项，全部通过、无跳过，最大绝对误差约 2.64e-11。使用固定 4 张训练图像、784→8→10 网络、float64、中心差分步长 1e-5；绝对容差为 1e-7，相对容差为 1e-5，跨越 ReLU 折点时跳过。

混淆矩阵和错误样例使用预先固定的种子 42。性能表和逐类召回率使用种子 42、43、44 的汇总。混淆矩阵行表示真实类别，列表示预测类别。错误图依据 Softmax 的六个最高计数有向混淆类别对，每对选取原始索引最小的样本。

## 环境配置

已验证环境：Apple M5、16 GB 内存、macOS arm64、Python 3.11.16、PyTorch 2.14.1、Torchvision 0.29.1、NumPy 2.4.6、Matplotlib 3.11.2。正式实验使用 CPU、4 线程。

安装 Git 和 Conda 后执行：

```bash
git clone https://github.com/yttt29/DeepLearning_Homework2.git
cd DeepLearning_Homework2
conda env create -f environment.yml
conda activate dlcv-lab1
python -m pip check
```

已有同名环境时可复用，或创建时指定其他环境名。`environment.yml` 提供依赖版本；`requirements-macos-arm64.lock.txt` 记录本次 macOS 环境，训练脚本也将其复制到本地运行记录中。其他平台从 `environment.yml` 安装，不保证跨平台逐位一致。

## 数据与配置

```bash
python scripts/prepare_data.py
```

首次运行从 Fashion-MNIST 作者的官方 GitHub 镜像下载数据，已有文件可复用。原始数据及压缩包约占 86 MB。

官方训练集 60,000 张按 `np.random.default_rng(42).permutation(60000)` 固定划分：前 50,000 张训练，后 10,000 张验证。脚本生成 `data/split_seed42.npz`；已有划分不一致时拒绝覆盖。官方测试集 10,000 张仅用于最终评价。

| 配置项 | 设置 |
|---|---|
| 预处理 | 像素除以 255，float32，展平为 784 维 |
| 三模型 | 784→10；784→256→10；784→256→ReLU→10 |
| 训练随机种子 | 42、43、44；数据划分种子固定为 42 |
| 梯度检查随机种子 | 初始化 42；检查位置抽样 43 |
| 初始化 | Xavier 均匀分布，gain=1，偏置为 0 |
| 批大小 / 轮数 | 128 / 30，保留末批 |
| 优化器 | SGD，学习率 0.01，动量 0.9 |
| 其他训练设置 | weight decay=0；无 Dropout、数据增强、学习率调度或提前停止 |
| 最佳模型选择 | 最低验证交叉熵，并列时取最早轮次 |
| 设备 | CPU，4 线程 |

`configs/baseline.json` 用于三模型主比较；`configs/diagnosis_lr003.json` 仅将两层线性模型的学习率降至 0.003，其余条件不变。同一种子下两个两层模型初始参数相同，每次运行重建数据加载器以保持批次顺序可比。

## 运行命令

以下命令均在项目根目录、激活环境并准备数据后执行。训练输出目录必须是新目录，程序拒绝覆盖已有运行。

### 1. 梯度检查与短程试跑

```bash
python scripts/numpy_mlp.py
python train.py --smoke --device cpu
```

梯度检查会更新 `results/gradient_check/`。短程试跑每模型最多 2 轮，每轮只取少量批次，用于检查流程，不用于报告性能比较。

### 2. 正式训练

```bash
python train.py --config configs/baseline.json --output results/training/reproduce_baseline
python train.py --config configs/diagnosis_lr003.json --output results/training/reproduce_diagnosis
```

基线为 3 模型 × 3 种子 × 30 轮，诊断为 1 模型 × 3 种子 × 30 轮。训练过程不读取测试集。每轮计算训练、验证指标，按验证损失保存最佳模型。

### 3. 最终评价

确认训练和诊断方案后执行：

```bash
python evaluate.py --run results/training/reproduce_baseline --split validation
python evaluate.py --run results/training/reproduce_diagnosis --split validation
python evaluate.py --run results/training/reproduce_baseline --split test
python evaluate.py --run results/training/reproduce_diagnosis --split test
```

评价读取新训练的 `best.pt`，生成性能汇总、混淆矩阵、逐类召回率、预测与错误样本索引。测试结果不用于继续调参或重新选择模型。仓库未附旧模型权重，需完成上一步训练才能执行评价。

### 4. 绘制新运行的学习曲线

```bash
python scripts/plot_learning_curves.py --baseline results/training/reproduce_baseline --diagnosis results/training/reproduce_diagnosis --out results/figures_reproduce
```

绘图脚本对应本实验的 30 轮和种子 42、43、44。报告中的训练曲线使用轮末固定模型在训练集上的 `train_eval_*` 指标，与同一模型的验证指标比较。中文图表需要 Arial Unicode MS、Noto Sans CJK SC、SimHei 或 Heiti TC 字体之一。

## 文件结构与提交检查

```text
实验报告.pdf                         待补充
README.md                            环境、运行命令、配置与随机种子
源代码：data.py / models.py / train.py / evaluate.py
environment.yml                      环境依赖
requirements-macos-arm64.lock.txt    本次环境版本记录
configs/                             基线与诊断配置
scripts/                             数据准备、梯度检查与学习曲线绘制
results/gradient_check/              梯度检查结果
results/summary/                     最终性能表、逐类召回率表
results/figures/                     四张最终图
```

- [x] 可运行源码与实验配置
- [x] README：环境、命令、配置、随机种子
- [x] 梯度检查结果、性能表、训练与验证曲线
- [x] 混淆矩阵、逐类召回率、典型错误
- [ ] PDF 实验报告

PDF 完成后以 `实验报告.pdf` 放入根目录，并更新本 README 的入口与状态。提交时使用仓库当前版本的文件；截止时间、入口和打包格式以课程平台通知为准。
