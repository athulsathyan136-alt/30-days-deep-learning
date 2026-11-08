
import torch
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

# Training
model.train()
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        predicted = outputs.logits.argmax(1)
        correct += (predicted == batch["labels"]).sum().item()
        total += len(batch["labels"])

print(f"Accuracy: {correct/total:.4f}")

# Save locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("✅ Model saved to ./model/")