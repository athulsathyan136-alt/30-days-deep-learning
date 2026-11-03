"""
Lesson 16: Transfer Learning with Pretrained ResNet-18 on MNIST
Adapts ImageNet-trained ResNet for digit classification.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import time


# ============================================================
# 1. Data (resize to 64×64 for faster training on CPU)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),          # ResNet needs bigger than 28×28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset for faster training (10,000 train, full test)
train_ds = Subset(train_ds, range(10000))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")


# ============================================================
# 2. Load Pretrained ResNet-18
# ============================================================
try:
    model = models.resnet18(weights='IMAGENET1K_V1')
    print("Loaded pretrained ResNet-18 (new API)")
except TypeError:
    model = models.resnet18(pretrained=True)
    print("Loaded pretrained ResNet-18 (old API)")

# ============================================================
# 3. Adapt ResNet for MNIST
# ============================================================

# Change first layer: accept 1-channel grayscale instead of 3-channel RGB
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Change last layer: 1000 ImageNet classes → 10 digit classes
model.fc = nn.Linear(512, 10)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 4. Loss + Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 5. Training loop (5 epochs)
# ============================================================
print("\nTraining...\n")
t0 = time.time()

for epoch in range(5):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in train_loader:
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (y_pred.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            y_pred = model(X)
            predicted = y_pred.argmax(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_acc = correct / total

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s")
print(f"✅ Final test accuracy: {test_acc:.4f}")