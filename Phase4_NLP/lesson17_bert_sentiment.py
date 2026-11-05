import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)

def tokenize(examples):
    return tokenizer(examples["text"],truncation=True,padding="max_length",max_length=128)

dataset =  dataset.map(tokenize,batched=True)
dataset = dataset.remove_columns(["text"])


collator = DataCollatorWithPadding(tokenizer)
train_ds = dataset["train"].select(range(2000))
test_ds = dataset["test"].select(range(2000))

train_loader = DataLoader(train_ds,batch_size=16,shuffle=True,collate_fn=collator)
test_loader = DataLoader(test_ds,batch_size=32,shuffle=False,collate_fn=collator)

optimizer = torch.optim.Adam(model.parameters(),lr=0.00002)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        output = model(**batch)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for batch in test_loader:
            output = model(**batch)
            predicted = output.logits.argmax(1)
            correct += (predicted==batch["labels"]).sum().item()
            total+=len(batch["labels"])

    accuracy = correct/total
    print(f"Epoch :{epoch+1} accuarcy :{accuracy:.4f}")
