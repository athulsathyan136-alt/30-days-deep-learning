import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),
                               transforms.Lambda(lambda x: x.view(-1))])

train_ds = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_ds = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_ds,batch_size=128,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=256,shuffle=False)

def make_model():
    return nn.Sequential(
        nn.Linear(784,256),nn.ReLU(),
        nn.Linear(256,128),nn.ReLU(),
        nn.Linear(128,64),nn.ReLU(),
        nn.Linear(64,10),
    )

def train(model,optimizer,epochs=5):
    criterion = nn.CrossEntropyLoss()
    losses,accs = [],[]

    for epoch in range(epochs):
        model.train()
        running_loss,total = 0.0,0
        for X,y in train_loader:
            logits = model(X)
            loss = criterion(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*X.size(0)
            total+=X.size(0)
        losses.append(running_loss/total)

        model.eval()
        correct,total = 0,0
        with torch.no_grad():
            for X,y in test_loader:
                logits = model(X)
                correct +=(logits.argmax(1)==y).sum().item()
                total+=X.size(0)
        accs.append(correct/total)
    return losses,accs

optimizers_config = {
    "SGD":           lambda p: torch.optim.SGD(p, lr=0.01),
    "SGD+Momentum":  lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "RMSprop":       lambda p: torch.optim.RMSprop(p, lr=0.001),
    "Adam":          lambda p: torch.optim.Adam(p, lr=0.001),
    "AdamW":         lambda p: torch.optim.AdamW(p, lr=0.001),
}

results = {}

for name ,opt_fn in optimizers_config.items():
    print(f"\nTraining with {name}...")
    torch.manual_seed(42)
    model = make_model()
    optimizer = opt_fn(model.parameters())
    t0 = time.time()
    losses,accs = train(model,optimizer,epochs=5)
    elapsed = time.time() -t0
    results[name] = (losses,accs,elapsed)
    print(f"  Final test acc: {accs[-1]:.4f} | Time: {elapsed:.1f}s")

print("\n" + "=" * 70)
print(f"{'Optimizer':<15} {'Final Test Acc':<20} {'Final Loss':<15} {'Time':<10}")
print("=" * 70)
for name, (losses, accs, elapsed) in results.items():
    print(f"{name:<15} {accs[-1]:<20.4f} {losses[-1]:<15.4f} {elapsed:<10.1f}")
print("=" * 70)

fig,axes = plt.subplots(1,2,figsize=(14,5))

colors = {
    "SGD": "gray",
    "SGD+Momentum": "orange",
    "RMSprop": "purple",
    "Adam": "blue",
    "AdamW": "green",
}

for name , (losses,accs,elapsed) in results.items():
    axes[0].plot(losses,label=name,color= colors[name],lw=2)
    axes[1].plot(accs,label=name,color= colors[name],lw=2)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Train Loss")
axes[0].set_title("Loss by Optimizer")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Accuracy by Optimizer")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=100)
plt.show()

print("\n✅ Done! Saved optimizer_comparison.png")