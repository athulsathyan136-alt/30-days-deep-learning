
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import copy


# ============ 1. Data with augmentation ============
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.Lambda(lambda x: x.view(-1))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")


# ============ 2. Model ============
class Fashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


model = Fashion()


# ============ 3. He initialization ============
def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_he)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


# ============ 4. Loss, optimizer, scheduler ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


# ============ 5. Evaluate ============
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            loss_sum += criterion(logits, y).item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)
    return loss_sum / total, correct / total


# ============ 6. Training ============
epochs = 20
train_losses, test_losses = [], []
train_accs, test_accs = [], []
lrs = []

best_val_acc = 0.0
best_model_state = None
best_epoch = -1

print("\nStarting training...\n")
t0 = time.time()

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in train_loader:
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    lrs.append(optimizer.param_groups[0]['lr'])

    scheduler.step()

    marker = ""
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        marker = " ⭐ best"

    print(f"Epoch {epoch+1:2d} | "
          f"Train: {train_loss:.4f}/{train_acc:.4f} | "
          f"Test: {test_loss:.4f}/{test_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s")
print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f}")


# ============ 7. Evaluate best model ============
model.load_state_dict(best_model_state)
test_loss, test_acc = evaluate(test_loader)
print(f"\n🏆 Best model test accuracy: {test_acc:.4f}")


# ============ 8. Save ============
torch.save(best_model_state, "fashion_model.pth")
print("✅ Saved fashion_model.pth")


# ============ 9. Per-class accuracy ============
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model.eval()
class_correct = [0] * 10
class_total = [0] * 10
confusion = torch.zeros(10, 10, dtype=torch.int64)

with torch.no_grad():
    for X, y in test_loader:
        preds = model(X).argmax(1)
        for t, p in zip(y, preds):
            confusion[t.item(), p.item()] += 1
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()] += 1

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    acc = class_correct[i] / class_total[i]
    print(f"  {name:<12} {acc:.4f}")


# ============ 10. Plot ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, 'b-', label='Train')
axes[0].plot(test_losses, 'r-', label='Test')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_accs, 'b-', label='Train')
axes[1].plot(test_accs, 'r-', label='Test')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True)

im = axes[2].imshow(confusion.numpy(), cmap="Blues")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True")
axes[2].set_title("Confusion Matrix")
axes[2].set_xticks(range(10))
axes[2].set_yticks(range(10))
axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
axes[2].set_yticklabels(class_names, fontsize=8)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("fashion_results.png", dpi=100)
plt.show()

print("\n✅ Done! Saved fashion_results.png")