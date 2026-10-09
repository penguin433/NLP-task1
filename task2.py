
import pandas as pd
from pathlib import Path
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from accuracyandmacro import accuracy, macro

# -------------------- 1. 读取 NYT 数据 --------------------

# 获取 task2.py 所在的项目目录
BASE_DIR = Path(__file__).resolve().parent

# NYT 数据集路径
NYT_PATH = BASE_DIR / "HW-1" / "nyt.csv"

# 读取 CSV 文件
df = pd.read_csv(NYT_PATH)

# 删除 text 或 label 为空的数据，并重新编号
df = df.dropna(
    subset=["text", "label"]
).reset_index(drop=True)

# 提取文本和标签
alltexts = df["text"]
labels = df["label"]




# -------------------- 2. 定义分词函数 --------------------

def tokenize_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text, preserve_line=True)
    return tokens


# -------------------- 3. 划分 NYT 数据集 --------------------

# 第一次：80% Training，20% Temporary
traintexts, temptexts, trainlabels, templabels = train_test_split(
    alltexts,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 第二次：把 Temporary 平均分成 Validation 和 Test
valtexts, testtexts, vallabels, testlabels = train_test_split(
    temptexts,
    templabels,
    test_size=0.5,
    random_state=42,
    shuffle=True
)



# -------------------- 4. 对 NYT 文本进行分词 --------------------

train_tokens = [tokenize_text(text) for text in traintexts]
val_tokens = [tokenize_text(text) for text in valtexts]
test_tokens = [tokenize_text(text) for text in testtexts]

EMBEDDING_DIM = 100

def document_to_vector(tokens, word_vectors):
    vectors = []
    for word in tokens:
        if word in word_vectors:
            vectors.append(word_vectors[word])
    if len(vectors) == 0:
        return np.zeros(EMBEDDING_DIM)
    document_vector = np.mean(vectors, axis=0)
    return document_vector
# -------------------- 多篇文档转换成矩阵 --------------------

def documents_to_matrix(tokenized_documents, word_vectors):
    document_vectors = []

    for tokens in tokenized_documents:
        vector = document_to_vector(tokens, word_vectors)
        document_vectors.append(vector)

    document_matrix = np.vstack(document_vectors)

    return document_matrix

# -------------------- Pre-trained GloVe --------------------

GLOVE_PATH = BASE_DIR / "glove" / "glove.6B.100d.txt"

def load_glove(path):
    glove_vectors = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            word, values = line.strip().split(" ", 1)
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            glove_vectors[word] = vector

    return glove_vectors

glove_vectors = load_glove(GLOVE_PATH)

train_glove = documents_to_matrix(train_tokens, glove_vectors)
val_glove = documents_to_matrix(val_tokens, glove_vectors)
test_glove = documents_to_matrix(test_tokens, glove_vectors)
# -------------------- GloVe Logistic Regression --------------------

# -------------------- GloVe 模型选择 --------------------

best_macro = -1
best_C = None
glove_model = None

for C in [0.1, 1, 10]:

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(train_glove, trainlabels)

    val_predict = model.predict(val_glove)
    val_score = macro(vallabels, val_predict)

    print("C =", C, "Validation Macro-F1:", val_score)

    if val_score > best_macro:
        best_macro = val_score
        best_C = C
        glove_model = model

print("Best C:", best_C)

print("training complete")

# -------------------- GloVe Validation --------------------

val_glove_predict = glove_model.predict(val_glove)

val_glove_accuracy = accuracy(vallabels, val_glove_predict)
val_glove_macro = macro(vallabels, val_glove_predict)

print("GloVe Validation Accuracy:", val_glove_accuracy)
print("GloVe Validation Macro-F1:", val_glove_macro)


# -------------------- GloVe Test --------------------

test_glove_predict = glove_model.predict(test_glove)

test_glove_accuracy = accuracy(testlabels, test_glove_predict)
test_glove_macro = macro(testlabels, test_glove_predict)

print("GloVe Test Accuracy:", test_glove_accuracy)
print("GloVe Test Macro-F1:", test_glove_macro)