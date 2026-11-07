import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


dataset = load_dataset("stanfordnlp/imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])

collator = DataCollatorWithPadding(tokenizer)

train_ds = dataset["train"].shuffle(seed=42).select(range(1000))
test_ds = dataset["test"].shuffle(seed=42).select(range(500))

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collator)


optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)


model.train()
for batch in train_loader:
    output = model(**batch)
    loss = output.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        output = model(**batch)
        predict = output.logits.argmax(1)
        correct += (predict == batch["labels"]).sum().item()
        total += len(batch["labels"])

accuracy = correct / total

# ============ DEBUG ============
from collections import Counter

print("\n=== Test set distribution ===")
test_labels = [test_ds[i]["label"] for i in range(len(test_ds))]      # ← test_ds
print(f"Test labels: {Counter(test_labels)}")

train_labels = [train_ds[i]["label"] for i in range(len(train_ds))]   # ← train_ds
print(f"Train labels: {Counter(train_labels)}")


model.eval()
pred_labels = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        predictions = outputs.logits.argmax(1)
        pred_labels.extend(predictions.tolist())

print(f"\nPredicted labels: {Counter(pred_labels)}")


print("\n=== First 10 test samples ===")
for i in range(10):
    print(f"Sample {i}: true={test_ds[i]['label']}")                  # ← test_ds


print(f"\nAccuracy: {accuracy:.4f}")


# ============ SAVE ============
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("✅ Model saved to ./model/")