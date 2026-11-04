import torch 
from torchvision import datasets,transforms,models
from torch.utils.data import Subset,DataLoader
import torch.nn as nn
import time

transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

train_ds = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_ds = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_ds = Subset(train_ds,range(5000))

train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=128,shuffle=False)

model = models.resnet18(weights='IMAGENET1K_V1')


model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512,10)

critersion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.001)

for epoch in range(5):
    model.train()
    for X,y in train_loader:
        y_pred = model(X)
        loss = critersion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X,y in test_loader:
            y_pred = model(X)
            predict = y_pred.argmax(1)
            correct += (predict==y).sum().item()
            total+=X.size(0)

    accuaray = correct/total
    print(f"epoch {epoch+1}: accuarcy : {accuaray:.4f}")