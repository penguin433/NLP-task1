
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import math
from torch.utils.data import TensorDataset, DataLoader
from accuracyandmacro import accuracy, macro


# -------------------- 1. 加载BERT --------------------

MODEL_NAME = "google-bert/bert-base-uncased"
torch.manual_seed(42)

# 检查训练设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 加载BERT分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 加载预训练BERT和三分类层
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

print("BERT model loaded!")


# -------------------- 2. 读取 NYT 数据 --------------------

BASE_DIR = Path(__file__).resolve().parent
NYT_PATH = BASE_DIR / "HW-1" / "nyt.csv"

df = pd.read_csv(NYT_PATH)
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

alltexts = df["text"]
labels = df["label"]


# -------------------- 3. 划分数据集 --------------------

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

print("Training Set:", len(traintexts))
print("Validation Set:", len(valtexts))
print("Test Set:", len(testtexts))

# -------------------- 4. 标签转换 --------------------

label_names = sorted(labels.unique())
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

train_y = trainlabels.map(label2id).tolist()
val_y = vallabels.map(label2id).tolist()
test_y = testlabels.map(label2id).tolist()


# -------------------- 5. BERT Tokenization --------------------

def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

train_encodings = encode_texts(traintexts)
val_encodings = encode_texts(valtexts)
test_encodings = encode_texts(testtexts)

print("Label mapping:", label2id)
print("Training input shape:", train_encodings["input_ids"].shape)
print("Validation input shape:", val_encodings["input_ids"].shape)
print("Test input shape:", test_encodings["input_ids"].shape)

# -------------------- 6. 准备 DataLoader --------------------

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_y, dtype=torch.long)
)

val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_y, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


# -------------------- 7. BERT Fine-tuning --------------------

model.config.label2id = label2id
model.config.id2label = id2label

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

best_macro = -1
SAVE_PATH = BASE_DIR / "models" / "bert_best"

for epoch in range(3):

    model.train()
    total_loss = 0

    for step, (ids, mask, y) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        outputs = model(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            labels=y.to(device)
        )

        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()

        if step % 200 == 0:
            print(f"Epoch {epoch+1}/3, Step {step}/{len(train_loader)}", flush=True)

    # -------------------- Validation --------------------

    model.eval()
    val_predictions = []

    with torch.no_grad():
        for ids, mask, y in val_loader:
            outputs = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device)
            )

            predictions = outputs.logits.argmax(dim=1)
            val_predictions.extend(predictions.cpu().tolist())

    val_accuracy = accuracy(val_y, val_predictions)
    val_macro = macro(val_y, val_predictions)

    print(f"Epoch {epoch+1} Average Loss:", total_loss / len(train_loader))
    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro-F1:", val_macro)

    # 保存验证集表现最好的模型
    if val_macro > best_macro:
        best_macro = val_macro
        model.save_pretrained(str(SAVE_PATH))
        tokenizer.save_pretrained(str(SAVE_PATH))
        print("Best BERT model saved!")

print("BERT Fine-tuning completed!")

