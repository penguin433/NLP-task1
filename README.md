
# NLP 文本分类实验

## 一、项目简介

本项目为自然语言处理课程的文本分类实验，使用 NYT 新闻数据集完成分类任务，主要包含以下三个部分：

- Task 1：使用 Binary Bag of Words 和 Word Frequency 进行文本分类。
- Task 2：使用预训练 GloVe、AG News 训练的 Word2Vec、NYT 训练的 Word2Vec 构建文档向量，并使用逻辑回归分类。
- Task 3：使用预训练 BERT 模型进行 Fine-tuning，完成新闻文本分类。

所有实验统一使用 NYT 数据集进行分类评价，采用 Accuracy 和 Macro-F1 作为评价指标。

## 二、项目文件说明

| 文件或目录              | 用途                                       |
| ----------------------- | ------------------------------------------ |
| `HW-1/nyt.csv`        | NYT 新闻分类数据集                         |
| `HW-1/ag.csv`         | AG News 词向量训练数据                     |
| `task1.py`            | Binary Bag of Words 和 Word Frequency 实验 |
| `task2.py`            | 预训练 GloVe 词向量实验                    |
| `task2-2.py`          | 使用 AG News 训练 Word2Vec                 |
| `task2-3.py`          | 使用 NYT 训练集训练 Word2Vec               |
| `task3.py`            | BERT 模型微调                              |
| `task3-test.py`       | 加载最佳 BERT 模型并进行测试               |
| `accuracyandmacro.py` | Accuracy 和 Macro-F1 评价函数              |
| `result/`             | 实验结果记录                               |
| `report/nlp1.pdf`     | 实验报告 PDF                               |

说明：大型预训练词向量和模型权重未上传至 GitHub，需要按照下文的步骤下载或生成。

## 三、运行环境及依赖安装

实验使用 Windows 操作系统、Visual Studio Code 和 Python，采用 Anaconda 管理 Python 环境。

### 1. Task 1 和 Task 2

Task 1 和 Task 2 使用原有的 Anaconda 环境。

在 VS Code 终端中执行：

```powershell
conda activate base
python -m pip install pandas
python -m pip install numpy
python -m pip install nltk
python -m pip install scikit-learn
python -m pip install gensim
```

### 2. Task 3

BERT 实验使用独立的 Python 3.11 环境。

创建环境并安装依赖：

```powershell
conda create -n nlp-bert python=3.11 -y
conda activate nlp-bert

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers
python -m pip install accelerate
python -m pip install pandas
python -m pip install numpy
python -m pip install scikit-learn
```

以上命令安装 CPU 版本的 PyTorch，适用于不使用 NVIDIA CUDA 显卡的运行环境。

`pathlib`、`time`、`math` 等属于 Python 标准库，无需单独安装。

## 四、模型和数据准备

### 1. 数据集

项目中的 `HW-1/` 目录包含：

- `nyt.csv`：用于新闻分类。
- `ag.csv`：用于训练 AG News Word2Vec。

NYT 数据按照 80%、10%、10% 的比例划分训练集、验证集和测试集，使用 `random_state=42`，保证各实验采用相同的数据划分。

### 2. GloVe 词向量

下载预训练 GloVe 词向量：

https://nlp.stanford.edu/data/glove.6B.zip

解压后，将 `glove.6B.100d.txt` 放置在：

```text
glove/glove.6B.100d.txt
```

### 3. Word2Vec 模型

运行 `task2-2.py` 或 `task2-3.py` 时，程序会检查相应模型是否存在。

如果模型不存在，则训练并保存到 `models/` 目录；如果模型已经存在，则直接加载。

### 4. BERT 模型

实验使用 Hugging Face 提供的预训练模型：

`google-bert/bert-base-uncased`

模型地址：

https://huggingface.co/google-bert/bert-base-uncased

首次运行需要能够访问 Hugging Face 以下载模型文件。

训练完成后，程序将验证集 Macro-F1 最高的模型保存到：

```text
models/bert_best/
```

## 五、程序运行说明

请先进入项目根目录，再执行以下命令。

### 1. Task 1：词袋模型

```powershell
conda activate base
python task1.py
```

程序输出 Binary Bag of Words 和 Word Frequency 的分类评价指标。

### 2. Task 2：GloVe 和 Word2Vec

```powershell
conda activate base

python task2.py
python task2-2.py
python task2-3.py
```

三个脚本分别执行预训练 GloVe、AG News Word2Vec 和 NYT Word2Vec 分类实验。

### 3. Task 3：BERT 微调

```powershell
conda activate nlp-bert
python task3.py
```

训练设置：

- 预训练模型：BERT-base-uncased
- 最大输入长度：64
- Batch Size：8
- Learning Rate：2e-5
- Epochs：3

程序每轮结束后计算验证集指标，并保存最佳模型。

训练完成后，执行：

```powershell
python task3-test.py
```

该脚本从 `models/bert_best/` 加载模型，并在 NYT 测试集上输出 Accuracy 和 Macro-F1。

注意：由于 GitHub 仓库不包含训练好的 BERT 权重，首次运行 `task3-test.py` 前需要先运行 `task3.py` 完成训练。

## 六、实验结果

各方法在 NYT Test Set 上的结果如下：

| 方法                | Accuracy | Macro-F1 |
| ------------------- | -------: | -------: |
| Binary Bag of Words |   0.9792 |   0.9426 |
| Word Frequency      |   0.9809 |   0.9496 |
| Pre-trained GloVe   |   0.9774 |   0.9403 |
| AG News Word2Vec    |   0.9722 |   0.9251 |
| NYT Word2Vec        |   0.9792 |   0.9469 |
| BERT Fine-tuning    |   0.9800 |   0.9501 |

BERT 的三轮微调在实验所用 CPU 环境下耗时约 1 小时 47 分钟。

实验过程、参数设置、结果分析与方法讨论详见：

**`report/nlp1.pdf`**
