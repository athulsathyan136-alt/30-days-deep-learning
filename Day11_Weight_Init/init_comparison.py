import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),transforms.Lambda(lambda x: x.view(-1))])

train_ds = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_ds = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_ds,batch_size=128,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=256,shuffle=False)

def make_model():
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10),
    )

def init_zero(m):
    if isinstance(m,nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        
def init_small(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,std=1e-3)
        nn.init.zeros_(m.bias)

def init_large(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,std=1.0)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        nn.init.zeros_(m.bias)

def train(model,epochs=5,lr=0.01):
    critersion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    losses,accs =[],[]

    for epoch in range(epochs):
        model.train()
        running_loss,total = 0.0,0
        for X, y in train_loader:
            logits = model(X)
            loss = critersion(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*X.size(0)
            total += X.size(0)
        losses.append(running_loss/total)

        model.eval()
        correct,total = 0,0
        with torch.no_grad():
            for X,y in test_loader:
                logits = model(X)
                correct += (logits.argmax(1)==y).sum().item()
                total+=X.size(0)
        accs.append(correct/total)
    return losses,accs

inits = {
     "Zero":   init_zero,
    "Small":  init_small,
    "Large":  init_large,
    "Xavier": init_xavier,
    "He":     init_he,
}
 
results = {}
for name,init_fn in inits.items():
    print(f"\nTraining with {name} init....")
    torch.manual_seed(42)
    model = make_model()
    model.apply(init_fn)
    t0 = time.time()
    losses,accs = train(model,epochs=5)
    elapsed = time.time()- t0
    results[name] = (losses,accs,elapsed)
    print(f"  Final test acc: {accs[-1]:.4f} | Time: {elapsed:.1f}s")

print("\n" + "=" * 65)
print(f"{'Init':<10} {'Final Test Acc':<20} {'Final Loss':<20}")
print("=" * 65)
for name, (losses, accs, elapsed) in results.items():
    print(f"{name:<10} {accs[-1]:<20.4f} {losses[-1]:<20.4f}")
print("=" * 65)

fig,axes = plt.subplots(1,2,figsize=(14,5))
colors = {"Zero": "gray", "Small": "orange", "Large": "red",
          "Xavier": "blue", "He": "green"}

for name, (losses,accs,elapsed) in results.items():
    axes[0].plot(losses,label=name,color=colors[name],lw=2)
    axes[1].plot(accs,label=name,color=colors[name],lw=2)

axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Train Loss")
axes[0].set_title("Loss by Init")
axes[0].legend()
axes[0].grid(True)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Accuracy by Init")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("init_comparison.png", dpi=100)
plt.show()

print("\n✅ Done! Saved init_comparison.png")