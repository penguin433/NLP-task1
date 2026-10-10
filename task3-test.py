
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accuracyandmacro import accuracy, macro

# -------------------- 1. 加载最佳BERT --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bert_best"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()

# -------------------- 2. 读取NYT测试集 --------------------

df = pd.read_csv(BASE_DIR / "HW-1" / "nyt.csv")
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

alltexts = df["text"]
labels = df["label"]

_, temptexts, _, templabels = train_test_split(
    alltexts, labels, test_size=0.2,
    random_state=42, shuffle=True
)

_, testtexts, _, testlabels = train_test_split(
    temptexts, templabels, test_size=0.5,
    random_state=42, shuffle=True
)

# -------------------- 3. BERT Tokenization --------------------

test_encodings = tokenizer(
    testtexts.tolist(),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"]
)

test_loader = DataLoader(test_dataset, batch_size=8)

# -------------------- 4. Test Prediction --------------------

predictions = []

with torch.no_grad():
    for ids, mask in test_loader:
        outputs = model(input_ids=ids, attention_mask=mask)
        preds = outputs.logits.argmax(dim=1)
        predictions.extend(preds.tolist())

# -------------------- 5. 最终评价 --------------------

test_y = testlabels.map(model.config.label2id).tolist()

print("BERT Test Accuracy:", accuracy(test_y, predictions))
print("BERT Test Macro-F1:", macro(test_y, predictions))