import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast
from torch.utils.data import Dataset

#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("fake_news_dataset.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["label"], test_size=0.2)

tokenizer = BertTokenizerFast.from_pretrained("bert")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)


class FNData(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).to(device)
        return item

train_dataset = FNData(train_encodings, list(train_labels))
val_dataset = FNData(val_encodings, list(val_labels))

print("Model Loading")
model = BertForSequenceClassification.from_pretrained("bert", num_labels=2)
model.to(device)
print("Model Loaded")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,  # Enable mixed precision for faster training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()

def evaluate():
    model.eval()
    predictions, true_labels = [], []
    for batch in val_dataset:
        inputs = {key: torch.tensor(value).unsqueeze(0) for key, value in batch.items() if key != "labels"}
        labels = batch["labels"].item()
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(preds)
        true_labels.append(labels)

    acc = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {acc:.4f}")

evaluate()

model.save_pretrained("bert-fake-news-model")
tokenizer.save_pretrained("bert-fake-news-model")
