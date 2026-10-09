"""
author: Ma Siyuan
2026/9/28
"""
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from accuracyandmacro import accuracy, macro

def tokenize_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text, preserve_line=True)
    return tokens

DATA1 = "HW-1/nyt.csv"


#-------------------读取数据-----------------------

df = pandas.read_csv(DATA1)

# 把 text 或 label 为空的数据删掉，然后重新给剩下的数据编号
df = df.dropna(
    subset=["text", "label"]
).reset_index(drop=True)


#-------------------分词函数-----------------------

def tokenize_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text,preserve_line=True)
    return tokens


texts = df.loc[0, "text"]
textstoken = tokenize_text(texts)
alltexts = df["text"]
labels = df["label"]


#-----------划分成train(训练集) val(验证集) test(测试集)-------------

traintexts, temptexts, trainlabels, templabels = train_test_split(
    alltexts,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

valtexts, testtexts, vallabels, testlabels = train_test_split(
    temptexts,
    templabels,
    test_size=0.5,
    random_state=42,
    shuffle=True
)


#-------------------Binary Bag of Words-----------------------

# binary=True让词只记录有没有出现：出现=1，不出现=0
binaryvectorizer = CountVectorizer(tokenizer=tokenize_text,token_pattern=None,binary=True)


trainbinary = binaryvectorizer.fit_transform(traintexts)
valbinary = binaryvectorizer.transform(valtexts)
testbinary = binaryvectorizer.transform(testtexts)




binarymodel = LogisticRegression(max_iter=1000)

# 使用训练集训练逻辑回归
binarymodel.fit(trainbinary, trainlabels)



# 用训练好的模型预测验证集
valpredict = binarymodel.predict(valbinary)

# 调用自己写的Accuracy和Macro-F1
valaccuracy = accuracy(vallabels, valpredict)
valmacro = macro(vallabels, valpredict)

print("Binary Validation Accuracy:", valaccuracy)
print("Binary Validation Macro-F1:", valmacro)




# binary=False：记录单词实际出现次数
frequencyvectorizer = CountVectorizer(
    tokenizer=tokenize_text,
    token_pattern=None,
    binary=False
)

# 用训练集建立词表，并转换成词频向量
trainfrequency = frequencyvectorizer.fit_transform(traintexts)

# 验证集和测试集使用训练集建立好的词表
valfrequency = frequencyvectorizer.transform(valtexts)
testfrequency = frequencyvectorizer.transform(testtexts)




frequencymodel = LogisticRegression(max_iter=1000)

# Word Frequency训练模型
frequencymodel.fit(trainfrequency, trainlabels)




valfrequencypredict = frequencymodel.predict(valfrequency)

valfrequencyaccuracy = accuracy(vallabels, valfrequencypredict)
valfrequencymacro = macro(vallabels, valfrequencypredict)

print("Accuracy:", valfrequencyaccuracy)
print("Macro-F1:", valfrequencymacro)