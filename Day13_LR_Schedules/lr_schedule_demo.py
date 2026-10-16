
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time


# ============ 1. Data ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)


# ============ 2. Model ============
def make_model():
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10),
    )


# ============ 3. Train with a scheduler ============
def train_with_scheduler(scheduler_type, epochs=10):
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    if scheduler_type == "none":
        scheduler = None
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    else:
        raise ValueError(scheduler_type)

    losses, accs, lrs = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, total = 0.0, 0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            total += X.size(0)
        losses.append(running_loss / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                logits = model(X)
                correct += (logits.argmax(1) == y).sum().item()
                total += X.size(0)
        accs.append(correct / total)

        lrs.append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

    return losses, accs, lrs


# ============ 4. Train each variant ============
schedules = ["none", "step", "cosine", "exponential"]
results = {}

for name in schedules:
    print(f"\nTraining with '{name}' schedule...")
    t0 = time.time()
    losses, accs, lrs = train_with_scheduler(name, epochs=10)
    elapsed = time.time() - t0
    results[name] = (losses, accs, lrs, elapsed)
    print(f"  Final test acc: {accs[-1]:.4f} | Time: {elapsed:.1f}s")


# ============ 5. Comparison table ============
print("\n" + "=" * 65)
print(f"{'Schedule':<15} {'Final Test Acc':<20} {'Final Loss':<15}")
print("=" * 65)
for name, (losses, accs, lrs, elapsed) in results.items():
    print(f"{name:<15} {accs[-1]:<20.4f} {losses[-1]:<15.4f}")
print("=" * 65)


# ============ 6. Plot ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = {"none": "gray", "step": "blue", "cosine": "red", "exponential": "green"}

for name, (losses, accs, lrs, elapsed) in results.items():
    axes[0].plot(losses, label=name, color=colors[name], lw=2)
    axes[1].plot(accs,   label=name, color=colors[name], lw=2)
    axes[2].plot(lrs,    label=name, color=colors[name], lw=2)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Train Loss")
axes[0].set_title("Loss by Schedule")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Accuracy by Schedule")
axes[1].legend()
axes[1].grid(True)

axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("LR Schedule Curves")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("lr_schedule_comparison.png", dpi=100)
plt.show()

print("\n✅ Done! Saved lr_schedule_comparison.png")