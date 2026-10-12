import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)), transforms.Lambda(lambda x : x.view(-1))])

train_ds = datasets.MNIST(root='./data',train=True,download=True,transform = transform)
test_ds = datasets.MNIST(root='./data',train=False,download=True,transform = transform)

train_loader = DataLoader(train_ds,batch_size=128,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=256,shuffle=False)

def make_no_model():
    return nn.Sequential(
        nn.Linear(784,256),nn.ReLU(),
        nn.Linear(256,128),nn.ReLU(),
        nn.Linear(128,64),nn.ReLU(),
        nn.Linear(64,10),
    )

def make_model():
    return nn.Sequential(
        nn.Linear(784,256),nn.BatchNorm1d(256),nn.ReLU(),
                nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
                nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),
                nn.Linear(64,10),

    )

def train(model,epochs=10,lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_losses ,test_accs =[],[]

    for epoch in range(epochs):
        model.train()
        running_loss,total = 0.0,0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            total +=X.size(0)
        train_losses.append(running_loss/total)

        model.eval()
        correct,total = 0,0
        with torch.no_grad():
            for X,y in test_loader:
                logits = model(X)
                correct += (logits.argmax(1)==y).sum().item()
                total+=X.size(0)
            test_accs.append(correct/total)
    return train_losses,test_accs

print("\n[1] Training WITHOUT BatchNorm...")
torch.manual_seed(42)
make_no_models = make_no_model()
t0 = time.time()
loss_no_bn , acc_no_bn = train(make_no_models)
time_no_bn = time.time() - t0

print("\n[2] Training WITH BatchNorm...")
torch.manual_seed(42)
make_models = make_model()
t0 = time.time()
loss_bn,acc_bn = train(make_models)
time_bn = time.time()-t0

print('\n' + '=' *65)
print(f"{'Model':<15} {'Final Test Acc':<20} {'Time (s)':<15}")
print('='*65)
print(f"{'No BN':<15} {acc_no_bn[-1]:<20.4f} {time_no_bn:<15.2f}")
print(f"{'With BN':<15} {acc_bn[-1]:<20.4f} {time_bn:<15.2f}")
print('='*65)

fig,axes = plt.subplots(1,2,figsize=(14,5))

axes[0].plot(loss_no_bn,'b-',label='No BN',lw=2)
axes[0].plot(loss_bn,'r-',label='With BN',lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Train Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(acc_no_bn,'b-',label='No BN',lw=2)
axes[1].plot(acc_bn,'r-',label='With BN',lw=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Test Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("bn_comparison.png",dpi=100)
plt.show()

print("\n✅ Done! Saved bn_comparison.png")