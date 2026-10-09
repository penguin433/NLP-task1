import pandas as pd
from pathlib import Path
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from accuracyandmacro import accuracy, macro
from gensim.models import Word2Vec

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

#-----------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#以上跟task2.py一模一样




# -------------------- AG News Word2Vec --------------------

MODEL_PATH = BASE_DIR / "models" / "ag_word2vec.model"
MODEL_PATH.parent.mkdir(exist_ok=True)

if MODEL_PATH.exists():
    # 如果模型已经存在，直接加载
    ag_word2vec = Word2Vec.load(str(MODEL_PATH))

else:
    # 如果模型不存在，就使用AG News训练
    ag_df = pd.read_csv(BASE_DIR / "HW-1" / "ag.csv")
    ag_df = ag_df.dropna(subset=["text"])

    ag_tokens = [tokenize_text(text) for text in ag_df["text"]]

    ag_word2vec = Word2Vec(
        sentences=ag_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        epochs=5,
        workers=1,
        seed=42
    )

    ag_word2vec.save(str(MODEL_PATH))
# -------------------- AG Word2Vec 文档向量 --------------------

train_ag = documents_to_matrix(train_tokens, ag_word2vec.wv)
val_ag = documents_to_matrix(val_tokens, ag_word2vec.wv)
test_ag = documents_to_matrix(test_tokens, ag_word2vec.wv)

print("Training matrix:", train_ag.shape)
print("Validation matrix:", val_ag.shape)
print("Test matrix:", test_ag.shape)


# -------------------- AG Word2Vec Logistic Regression --------------------

best_macro = -1
best_C = None
ag_model = None

for C in [0.1, 1, 10]:

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(train_ag, trainlabels)

    val_predict = model.predict(val_ag)
    val_score = macro(vallabels, val_predict)

    print("C =", C, "Validation Macro-F1:", val_score)

    if val_score > best_macro:
        best_macro = val_score
        best_C = C
        ag_model = model

print("Best C:", best_C)


# -------------------- AG Word2Vec Validation --------------------

val_ag_predict = ag_model.predict(val_ag)

val_ag_accuracy = accuracy(vallabels, val_ag_predict)
val_ag_macro = macro(vallabels, val_ag_predict)

print("AG Validation Accuracy:", val_ag_accuracy)
print("AG Validation Macro-F1:", val_ag_macro)


# -------------------- AG Word2Vec Test --------------------

test_ag_predict = ag_model.predict(test_ag)

test_ag_accuracy = accuracy(testlabels, test_ag_predict)
test_ag_macro = macro(testlabels, test_ag_predict)

print("AG Test Accuracy:", test_ag_accuracy)
print("AG Test Macro-F1:", test_ag_macro)
