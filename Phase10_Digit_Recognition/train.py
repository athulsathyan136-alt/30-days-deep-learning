
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Load MNIST
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")


# ============================================================
# 2. CNN Model
# ============================================================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 28 → 14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = DigitCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 3. Loss + Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 4. Training (5 epochs)
# ============================================================
epochs = 5

for epoch in range(epochs):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    
    train_acc = correct / total
    
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    
    test_acc = correct / total
    
    print(f"Epoch {epoch+1} | Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# ============================================================
# 5. Save
# ============================================================
torch.save(model.state_dict(), "digit_cnn.pth")
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
print("Model saved to digit_cnn.pth")