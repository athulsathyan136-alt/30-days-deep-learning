import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms



transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

train_ds = datasets.MNIST(root="./data",train=True,download=True,transform = transform)
test_ds = datasets.MNIST(root="./data",train=False,download=True,transform = transform)

train_loader = DataLoader(train_ds,batch_size=256,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=126,shuffle=False)

model = nn.Sequential(
    nn.Conv2d(1,32,3,padding=1), #32,28,28
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2), #32,14,14
    nn.Dropout(0.25),

    nn.Conv2d(32,64,3,padding=1), #64,14,14
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2), #64,7,7
    nn.Dropout(0.25),

    nn.Flatten(),
    nn.Linear(64*7*7,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128,10),

)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):
    model.train()
    for X, y in train_loader:
        y_pred = model(X)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            y_pred = model(X)
            predicted = y_pred.argmax(1)
            correct += (predicted==y).sum().item()
            total+=X.size(0)

    accuracy = correct/total

    print(f"Epoch : {epoch+1} accuracy : {accuracy}")