import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),])

train_ds = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_ds = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=64,shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)         # (batch, 6, 12, 12)
        x = torch.relu(self.conv2(x))      # (batch, 16, 8, 8)
        x = torch.max_pool2d(x, 2)         # (batch, 16, 4, 4)
        x = x.view(x.size(0), -1)          # (batch, 256)
        x = torch.relu(self.fc1(x))        # (batch, 120)
        x = torch.relu(self.fc2(x))        # (batch, 84)
        x = self.fc3(x)                    # (batch, 10)
        return x

model = LeNet5()
print(f"\nModel: LeNet-5")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterin = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def evaluate(loader):
    model.train()
    correct,total,loss_sum = 0,0,0.0
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            loss_sum += criterin(logits,y).item() * X.size(0)
            correct +=(logits.argmax(1)==y).sum().item()
            total+=X.size(0)
    return loss_sum/total,correct/total


epochs = 10
train_losses,test_losses = [],[]
train_accs,test_accs = [],[]

print("\nTraining...\n")
t0 = time.time()

for epoch in range(epochs):
    model.train()
    running_loss,correct,total = 0.0,0,0
    for X,y in train_loader:
        logits = model(X)
        loss = criterin(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*X.size(0)
        correct += (logits.argmax(1)==y).sum().item()
        total+=X.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    test_loss,test_acc = evaluate(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d} | "
          f"Train: {train_loss:.4f}/{train_acc:.4f} | "
          f"Test: {test_loss:.4f}/{test_acc:.4f}")

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s")
print(f"✅ Final test accuracy: {test_accs[-1]:.4f}")

print("\nExtracting feature maps from first conv layer...")

model.eval()
sample_img,_ = test_ds[0]
sample = sample_img.unsqueeze(0)

with torch.no_grad():
    conv1_out = torch.relu(model.conv1(sample))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Original image
axes[0, 0].imshow(sample_img.squeeze(), cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis('off')

# 6 feature maps
for i in range(6):
    row = (i + 1) // 4
    col = (i + 1) % 4
    axes[row, col].imshow(conv1_out[0, i].numpy(), cmap='viridis')
    axes[row, col].set_title(f"Filter {i+1}")
    axes[row, col].axis('off')

# Loss/accuracy curves in last slot
axes[1, 3].plot(train_losses, 'b-', label='Train')
axes[1, 3].plot(test_losses, 'r-', label='Test')
axes[1, 3].set_xlabel("Epoch")
axes[1, 3].set_ylabel("Loss")
axes[1, 3].set_title("Loss")
axes[1, 3].legend()
axes[1, 3].grid(True)

plt.tight_layout()
plt.savefig("lenet_results.png", dpi=100)
plt.show()

print("✅ Saved lenet_results.png")
print("\n✅ Done!")