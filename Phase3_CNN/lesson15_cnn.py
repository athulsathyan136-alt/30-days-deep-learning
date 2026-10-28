import torch 
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])


train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1,6,5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(6,16,5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(10):
    model.train()
    for X,y in train_loader:
        y_pred = model(X)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X,y in test_loader:
            y_pred = model(X)
            prediced = y_pred.argmax(1)
            correct += (prediced == y).sum().item()
            total += X.size(0)
    accuarcy = correct/total
    print(f"epoch {epoch+1}:  accuary {accuarcy}")