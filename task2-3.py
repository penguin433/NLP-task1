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


# -------------------- NYT Word2Vec --------------------

MODEL_PATH = BASE_DIR / "models" / "nyt_word2vec.model"
MODEL_PATH.parent.mkdir(exist_ok=True)

if MODEL_PATH.exists():
    nyt_word2vec = Word2Vec.load(str(MODEL_PATH))

else:
    nyt_word2vec = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        epochs=5,
        workers=1,
        seed=42
    )

    nyt_word2vec.save(str(MODEL_PATH))

print("NYT Word2Vec ready!")
# -------------------- NYT Word2Vec 文档向量 --------------------

train_nyt = documents_to_matrix(train_tokens, nyt_word2vec.wv)
val_nyt = documents_to_matrix(val_tokens, nyt_word2vec.wv)
test_nyt = documents_to_matrix(test_tokens, nyt_word2vec.wv)

print("Training matrix:", train_nyt.shape)
print("Validation matrix:", val_nyt.shape)
print("Test matrix:", test_nyt.shape)
# -------------------- NYT Logistic Regression --------------------

best_macro = -1
best_C = None
nyt_model = None

for C in [0.1, 1, 10]:

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(train_nyt, trainlabels)

    val_predict = model.predict(val_nyt)
    val_score = macro(vallabels, val_predict)

    print("C =", C, "Validation Macro-F1:", val_score)

    if val_score > best_macro:
        best_macro = val_score
        best_C = C
        nyt_model = model

print("Best C:", best_C)
# -------------------- NYT Word2Vec Evaluation --------------------

val_predict = nyt_model.predict(val_nyt)
test_predict = nyt_model.predict(test_nyt)

print("NYT Validation Accuracy:", accuracy(vallabels, val_predict))
print("NYT Validation Macro-F1:", macro(vallabels, val_predict))

print("NYT Test Accuracy:", accuracy(testlabels, test_predict))
print("NYT Test Macro-F1:", macro(testlabels, test_predict))
