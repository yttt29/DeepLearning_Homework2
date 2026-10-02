# 从线性分类器到多层感知机

《深度学习与计算机视觉》课程实验。徐义涛，U202314356，种子2301班。

本项目在 Fashion-MNIST 上完成 Softmax、两层无激活线性网络和 ReLU MLP 的受控比较，包含 NumPy 反向传播及梯度检查、三模型训练、最终测试评价和学习率单因素诊断。

## 1. 阅读入口与完成状态

本仓库保存源代码、配置和正式实验结果。训练、单因素诊断及最终测试评价均已完成；**PDF 实验报告待补充，当前尚未完成最终提交。**

| 要求的实验内容 | 仓库中的结果入口 |
|---|---|
| 梯度检查 | [汇总](results/gradient_check/report.json)、[逐项明细](results/gradient_check/details.csv) |
| 三模型性能表 | [12 次运行明细](results/analysis/final_performance.csv)，三模型均值见下表 |
| 训练与验证曲线 | [三模型学习曲线](results/figures/baseline_learning_curves.png) |
| 混淆矩阵 | [三模型测试集混淆矩阵](results/figures/test_confusion_matrices.png) |
| 逐类召回率 | [三种子汇总](results/analysis/report_data.json)，逐次结果见各模型的 `test_evaluation/per_class.csv` |
| 典型错误 | [六个样例与配对预测](results/figures/test_error_examples.png) |
| 单因素训练诊断 | [对照曲线](results/figures/diagnosis_learning_curves.png)、[配对结果](results/analysis/diagnosis_paired.csv) |

图像与数值数据可直接查看，无需先运行代码。Markdown 报告源文件留在本地，正式报告将以 PDF 格式补入仓库。

主要测试结果如下。“±”为三个种子的样本标准差，单位为百分点，不是置信区间。

| 模型 | 学习率 | 测试准确率 |
|---|---:|---:|
| Softmax | 0.01 | 84.33% ± 0.11 |
| 两层线性，无激活 | 0.01 | 84.30% ± 0.20 |
| ReLU MLP | 0.01 | 88.15% ± 0.20 |
| 两层线性诊断 | 0.003 | 84.36% ± 0.14 |

## 2. 环境配置

已验证环境：Apple M5、16 GB 内存、macOS 26.6 arm64、Python 3.11.16、PyTorch 2.14.1、Torchvision 0.29.1、NumPy 2.4.6、Matplotlib 3.11.2。正式实验使用 CPU，4 线程；不依赖 CUDA 或 MPS。

先获取项目，再创建并激活环境（需要已安装 Git 和 Conda）：

```bash
git clone https://github.com/yttt29/DeepLearning_Homework2.git
cd DeepLearning_Homework2
conda env create -f environment.yml
conda activate dlcv-lab1
python -m pip check
```

`environment.yml` 包含实验依赖及版本。已有同名环境时可复用，或在创建命令中加 `-n 其他环境名` 后激活对应环境。

精确环境锁文件 `conda-osx-arm64.lock.txt` 和 `requirements-macos-arm64.lock.txt` 仅供兼容的 macOS arm64 环境使用。其他平台应从 `environment.yml` 解析依赖，不能直接复制 macOS 环境目录；本项目未验证其他平台的逐位一致性。

## 3. 数据与实验配置

先获取数据并检查固定划分：

```bash
python scripts/prepare_data.py
```

脚本从 Fashion-MNIST 作者的官方 GitHub 镜像下载数据，已有数据会复用。首次下载需要联网。官方训练集 60,000 张固定分为训练 50,000 张、验证 10,000 张；官方测试集 10,000 张只用于最终评价。下载和检查测试文件形状不属于使用测试指标选择模型。

划分使用 `np.random.default_rng(42).permutation(60000)`，前 50,000 个索引为训练集，其余为验证集，保存至 `data/split_seed42.npz`。已有划分若与规则不一致会报错，避免覆盖。原始数据及压缩包约占 86 MB；可通过脚本获取，不必随作业重复提交，但建议保留固定划分文件。

| 配置项 | 基线设置 |
|---|---|
| 输入 | 展平为 784 维，float32，像素除以 255 |
| 训练种子 | 42、43、44；划分种子始终为 42 |
| 三模型 | 784→10；784→256→10；784→256→ReLU→10 |
| 初始化 | Xavier 均匀分布，gain=1，偏置 0 |
| 批大小 / 轮数 | 128 / 30，保留末批 |
| 优化器 | SGD，学习率 0.01，动量 0.9 |
| 正则化与调度 | weight decay=0；无 Dropout、数据增强、学习率调度或提前停止 |
| 模型选择 | 最低验证交叉熵，若并列取最早轮次 |
| 设备 | CPU，4 线程 |

配置位于 `configs/baseline.json`。诊断配置 `configs/diagnosis_lr003.json` 仅选择两层线性模型，并将其学习率改为 0.003，其他训练条件不变。同一种子下两个两层模型初始参数相同；每次运行重建数据加载器，保证批次顺序可比较。

## 4. 从头运行

以下命令均在项目根目录、激活环境后执行。`reproduce_baseline` 和 `reproduce_diagnosis` 是新输出目录；如果已经存在，请另取目录名。程序拒绝覆盖已有训练或评价结果。

### 4.1 实现检查与短程试跑

```bash
python scripts/numpy_mlp.py
python -m unittest discover -s tests -v
python train.py --smoke --device cpu
```

NumPy 检查使用固定 4 张训练图像、784→8→10 网络、float64、初始化种子 42、检查抽样种子 43、中心差分步长 1e-5；每组最多抽查 20 项。绝对容差为 1e-7、相对容差为 1e-5，跨越 ReLU 折点时跳过。该命令会更新梯度检查汇总与明细。

本次记录为 58 项全部通过、无跳过，最大绝对误差约 2.64e-11。10 项实现检查均通过。短程试跑只检查执行流程，不用于报告中的性能比较。

### 4.2 正式训练与验证

```bash
python train.py --config configs/baseline.json --output results/training/reproduce_baseline
python train.py --config configs/diagnosis_lr003.json --output results/training/reproduce_diagnosis
python evaluate.py --run results/training/reproduce_baseline --split validation
python evaluate.py --run results/training/reproduce_diagnosis --split validation
python scripts/audit_training.py --run results/training/reproduce_baseline
python scripts/audit_training.py --run results/training/reproduce_diagnosis
```

基线为 3 模型 × 3 种子 × 30 轮，诊断为 1 模型 × 3 种子 × 30 轮。训练过程不读取测试集。每轮保存训练、验证及固定轮末模型的训练集指标，最佳模型按验证损失保存为 `best.pt`。

### 4.3 固定方案后的最终测试评价

```bash
python evaluate.py --run results/training/reproduce_baseline --split test
python evaluate.py --run results/training/reproduce_diagnosis --split test
```

测试评价使用已选定的 `best.pt`，不更新参数，也不根据测试表现重新选择模型。每个模型目录生成 `test_evaluation/`，保存预测、混淆矩阵、逐类召回率与错误样本索引；运行根目录生成 `test_performance.csv` 和 `test_aggregate.json`。

### 4.4 新运行的学习曲线

```bash
python scripts/plot_learning_curves.py --baseline results/training/reproduce_baseline --diagnosis results/training/reproduce_diagnosis --out results/figures_reproduce
```

绘图脚本针对本实验的 30 轮、种子 42/43/44 配置。中文字体优先使用 Arial Unicode MS、Noto Sans CJK SC、SimHei 或 Heiti TC；若系统没有这些字体，中文图中文字可能缺失，应使用包含中文字形的字体。

## 5. 复核本报告已有结果

报告对应原运行目录：

- `results/training/baseline_cpu_20261002/`：三模型主比较，9 次运行。
- `results/training/diagnosis_linear2_lr003_20261002/`：降低学习率诊断，3 次运行。

保留这两个完整目录和 `results/analysis/` 中的计划文件，即可在下载数据后复核现有结果，无需重新训练：

```bash
python scripts/audit_training.py --run results/training/baseline_cpu_20261002
python scripts/audit_training.py --run results/training/diagnosis_linear2_lr003_20261002
python scripts/analyze_diagnosis.py
python scripts/plot_learning_curves.py --diagnosis results/training/diagnosis_linear2_lr003_20261002
python scripts/prepare_report_assets.py
```

这些命令会重写分析汇总和图像，不改动模型权重或逐轮日志。`analyze_diagnosis.py` 与 `prepare_report_assets.py` 读取本次已保存计划中的目录及哈希，用于核对原报告；它们不能直接用于不同目录的新训练。新训练按第 4 节获得全部预测数据及学习曲线，若另做完整分析，应另存对应计划与输出，避免覆盖本报告证据。

`prepare_report_assets.py` 核对官方测试标签、模型文件哈希、预测与混淆矩阵，并独立重算交叉熵；随后生成 `report_data.json`、`final_performance.csv`、混淆矩阵图及错误样例图。

图中种子固定为 42，在测试评价前选定；统计表使用全部三个种子。错误图按 Softmax 的六个最高计数有向混淆类别对选取，每对取原始索引最小的样本，不按 MLP 能否纠正错误挑选。

## 6. 文件结构与指标说明

```text
实验报告.pdf                 待补充；当前仓库尚无该文件
README.md                    环境、运行及复核说明
requirements-macos-arm64.lock.txt
conda-osx-arm64.lock.txt
environment.yml              依赖配置
data.py                      数据加载与固定划分读取
models.py                    三模型及受控初始化
train.py                     共用训练与验证流程
evaluate.py                  验证/测试预测与指标导出
configs/                     基线及诊断配置
scripts/                     梯度检查、数据准备、审核、分析与绘图
tests/                       10 项实现检查
data/split_seed42.npz        固定划分索引
results/gradient_check/      梯度检查结果
results/figures/             报告图表
results/training/            正式运行日志、模型及逐样本评价
results/analysis/            诊断计划、汇总数据和评价计划
```

每次训练保存 `manifest.json`、源码快照、实际配置与汇总；每个模型/种子保存 `history.csv`、`best.pt` 和 `last.pt`。模型文件包含权重与选择元数据，不是完整断点续训状态。

- `train_loss/accuracy`：训练时逐批更新参数过程中累计的指标。
- `train_eval_loss/accuracy`：该轮结束时，固定模型在全部训练集上的指标。
- `val_loss/accuracy`：同一个轮末模型在验证集上的指标。
- 报告的训练曲线使用 `train_eval_*`，保证与验证曲线对应同一组参数。
- `predictions.npz`：原始样本索引、真实标签、预测、logits 和概率。
- `confusion_matrix.csv`：行是真实类别，列是预测类别。
- `per_class.csv`：每类样本数、正确数与召回率。
- `errors.csv`：错误样本索引、真实/预测标签及预测置信度。

训练清单中的 `test_set_used=false` 指训练阶段未使用测试集。最终测试评价在训练与诊断完成后单独执行，记录在 `final_evaluation_plan.json` 和各模型的 `test_evaluation/`，没有改写原训练清单。

## 7. 提交材料核对

以下六项实验内容均已完成，并已编入本地报告稿；第 1 节提供了当前可直接查看的结果入口。PDF 版待补充，届时应直接呈现全部图表及分析：

| 实验要求 | 报告位置 | 配套证据 |
|---|---|---|
| 梯度检查结果 | 3.3 节 | `results/gradient_check/report.json`、`details.csv` |
| 三模型性能表 | 4.1 节、附录 A | `results/analysis/final_performance.csv` |
| 训练与验证曲线 | 4.2 节图 1 | `results/figures/baseline_learning_curves.png`、各次 `history.csv` |
| 混淆矩阵 | 5.1 节图 2 | `test_confusion_matrices.png`、各次 `confusion_matrix.csv` |
| 逐类召回率 | 5.2 节 | 各次 `per_class.csv`、`results/analysis/report_data.json` |
| 至少一组典型错误 | 5.3 节图 3，6 张图片 | `test_error_examples.png`、逐样本预测与索引 |

单因素诊断见第 6 节与图 4，配置、训练前计划及配对结果均已保存。

正式提交建议保留原目录结构，包含：

1. PDF 实验报告（**目前缺失**；Markdown 不能替代要求中的 PDF）。
2. 本 README、`environment.yml`、两个环境锁文件、4 个根目录 Python 源码、`configs/`、`scripts/` 与 `tests/`。
3. `data/split_seed42.npz`；完整原始数据可以不附，按第 3 节下载。
4. `results/gradient_check/`、`results/figures/`、上述两个正式训练目录的完整内容。
5. `results/analysis/` 中的 `diagnosis_plan.json`、`diagnosis_results.json`、`diagnosis_metrics.csv`、`diagnosis_paired.csv`、`final_evaluation_plan.json`、`final_performance.csv`、`report_data.json`。
6. `results/device_benchmark/report.json`，用于核对报告中的设备计时。Markdown 报告源文件不在本次上传范围内。

源代码、README 和 PDF 是明确要求的提交文件类型；模型权重与完整预测数据是便于复核的建议附件，不是课程说明额外指定的硬性格式。建议保留它们以支持第 5 节的直接复核。当前正式材料总量约 37 MB，未包含未来 PDF，也未计 Git 压缩效果。

不需打包 Conda 环境、原始数据压缩包、`.vscode/`、`__pycache__/`、`.DS_Store`、`results/smoke/`、`results/setup/`、`results/analysis/qa/` 或临时预览文件。仓库 `.gitignore` 按提交范围限制跟踪文件；以后新增正式材料时，需要同步更新允许列表。

截止时间、提交入口、压缩包格式和大小限制以课程平台通知为准；实验说明没有指定统一文件名或必须使用 ZIP。若平台允许压缩包，可使用“U202314356_徐义涛_线性模型与MLP”等易识别名称。

## 8. 维护约定与完成情况

- 保留原始正式运行记录，后续实验使用新输出目录，不覆盖本报告结果。
- 新增或修改配置、命令、图表时，同步更新本 README。
- PDF 完成后以 `实验报告.pdf` 放在根目录，补充下载入口，并更新待办状态。
- 不依据本次测试结果继续调参；扩展实验应单独记录研究问题和评价方案。

- [x] 环境、数据与统一配置
- [x] NumPy 前向、反向与梯度检查
- [x] 数据加载、三模型与共用训练流程
- [x] CPU/MPS 测速与正式训练
- [x] 单因素诊断与最终测试评价
- [x] 性能表、学习曲线、混淆矩阵、逐类召回率与错误样例
- [x] Markdown 报告、面向读者的 README 与提交材料核对
- [ ] PDF 报告排版与最终打包检查
