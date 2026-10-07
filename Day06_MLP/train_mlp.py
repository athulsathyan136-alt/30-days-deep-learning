import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),
                                transforms.Lambda(lambda x:x.view(-1))])

train_ds = datasets.MNIST(root='./data',train=True,download=True,transform = transform)
test_ds = datasets.MNIST(root='./data',train=False,download=True,transform = transform)

train_loader = DataLoader(train_ds,batch_size= 64,shuffle=True)
test_loader = DataLoader(test_ds,batch_size= 64,shuffle=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
    def forward(self,x):
        return self.net(x)


model = MLP()
total_params = sum(p.numel() for p in model.parameters())
print(f'\nMODEL:\n{model}')
print(f'Total parameters : {total_params}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= 0.001)

def evaluate(loader):
    model.eval()
    correct,total,loss_sum = 0,0,0.0
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            loss_sum += criterion(logits,y).item()*X.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total +=X.size(0)
    return loss_sum/ total,correct/total

epochs = 10
train_losses,test_losses = [],[]
train_accs,test_accs =[],[]

for epoch in range(epochs):
    model.train()
    running_loss,correct,total = 0.0,0,0
    for X,y in train_loader:
        logits = model(X)
        loss = criterion(logits,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1)==y).sum().item()
        total+=X.size(0)

    train_loss = running_loss/total
    train_acc = correct/total
    test_loss,test_acc = evaluate(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d} | "f"Train Loss {train_loss:.4f} | Train Acc : {train_acc:.4f} | "f"Test Loss : {test_loss:.4f} | Test Acc :{test_acc:.4f}")

model.eval()
class_correct = [0]*10
class_total = [0]*10
confusion = torch.zeros(10,10,dtype=torch.int64)

with torch.no_grad():
    for X,y in test_loader:
        preds = model(X).argmax(1)

        for t,p in zip(y,preds):
            confusion[t.item(),p.item()] += 1
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()]+=1
print('\nPer-class accuracy: ')
for i in range(10):
    acc = class_correct[i] / class_total[i]
    print(f' Digit {i}: {acc:.4f}')


fig,axes = plt.subplots(1,3,figsize=(15,4))

axes[0].plot(train_losses,label='Train')
axes[0].plot(test_losses, label="Test")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_accs, label="Train")
axes[1].plot(test_accs, label="Test")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True)

im = axes[2].imshow(confusion.numpy(), cmap="Blues")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True")
axes[2].set_title("Confusion Matrix")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("mlp_results.png", dpi=100)
plt.show()

torch.save(model.state_dict(), "mlp_model.pth")
print(f"\n✅ Done! Final test accuracy: {test_accs[-1]:.4f}")
print("Saved mlp_results.png and mlp_model.pth")