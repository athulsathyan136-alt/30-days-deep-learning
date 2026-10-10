
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============ 1. Load small subset ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
full_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

torch.manual_seed(42)
idx = torch.randperm(len(full_train))[:500]
small_train = Subset(full_train, idx)

train_loader = DataLoader(small_train, batch_size=32, shuffle=True)
test_loader  = DataLoader(full_test, batch_size=256, shuffle=False)

print(f"Train: {len(small_train)} | Test: {len(full_test)}")


# ============ 2. Model factory ============
def make_model(use_dropout=False):
    layers = [nn.Linear(784, 512), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers += [nn.Linear(512, 256), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)


# ============ 3. Training ============
def train(model, weight_decay=0.0, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

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

        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for X, y in test_loader:
                logits = model(X)
                loss_sum += criterion(logits, y).item() * X.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += X.size(0)

        test_loss = loss_sum / total
        test_acc = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return train_losses, test_losses, train_accs, test_accs


# ============ 4. Train 3 variants ============
print("\n[1] Baseline (no regularization)")
torch.manual_seed(42)
base = make_model(use_dropout=False)
b_tr_loss, b_te_loss, b_tr_acc, b_te_acc = train(base)

print("\n[2] With Dropout (p=0.5)")
torch.manual_seed(42)
drop = make_model(use_dropout=True)
d_tr_loss, d_te_loss, d_tr_acc, d_te_acc = train(drop)

print("\n[3] With Weight Decay (λ=1e-2)")
torch.manual_seed(42)
wd = make_model(use_dropout=False)
w_tr_loss, w_te_loss, w_tr_acc, w_te_acc = train(wd, weight_decay=1e-2)


# ============ 5. Print comparison ============
print("\n" + "=" * 65)
print(f"{'Model':<18} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10}")
print("=" * 65)

for name, tr, te in [
    ("Baseline",    b_tr_acc[-1], b_te_acc[-1]),
    ("Dropout",     d_tr_acc[-1], d_te_acc[-1]),
    ("WeightDecay", w_tr_acc[-1], w_te_acc[-1]),
]:
    gap = tr - te
    print(f"{name:<18} {tr:<12.4f} {te:<12.4f} {gap:<10.4f}")

print("=" * 65)
print("Bigger gap = more overfitting\n")


# ============ 6. Visualize ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(b_tr_acc, 'b--', label='Baseline train', alpha=0.5)
axes[0].plot(b_te_acc, 'b-',  label='Baseline test',  lw=2)
axes[0].plot(d_tr_acc, 'g--', label='Dropout train',  alpha=0.5)
axes[0].plot(d_te_acc, 'g-',  label='Dropout test',   lw=2)
axes[0].plot(w_tr_acc, 'r--', label='WeightDecay train', alpha=0.5)
axes[0].plot(w_te_acc, 'r-',  label='WeightDecay test',  lw=2)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy — Bigger Gap = More Overfitting")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(b_te_loss, 'b-', label='Baseline test', lw=2)
axes[1].plot(d_te_loss, 'g-', label='Dropout test', lw=2)
axes[1].plot(w_te_loss, 'r-', label='WeightDecay test', lw=2)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Loss")
axes[1].set_title("Test Loss — Lower = Better")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("overfit_comparison.png", dpi=100)
plt.show()

print("✅ Done! Saved overfit_comparison.png")