
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Data transforms
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")


# ============================================================
# 2. Model
# ============================================================
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


# ============================================================
# 3. Loss + Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 4. Training + Evaluation
# ============================================================
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (y_pred.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            y_pred = model(X)
            correct += (y_pred.argmax(1) == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# ============================================================
# 5. Per-class accuracy
# ============================================================
model.eval()
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for X, y in test_loader:
        preds = model(X).argmax(1)
        for t, p in zip(y, preds):
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()] += 1

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    acc = class_correct[i] / class_total[i]
    print(f"  {name:<12} {acc:.4f}")