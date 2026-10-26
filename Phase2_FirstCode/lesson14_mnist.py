import torch 
import torch.nn as nn
from  torch.utils.data import DataLoader
from torchvision import datasets, transforms 


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),transforms.Lambda (lambda x:x.view(-1))])

train_ds = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_ds = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=64,shuffle=False)


model = nn.Sequential(nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

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
            prediction = y_pred.argmax(1)
            corrected_predict = (prediction == y)
            correct += corrected_predict.sum().item()
            total += y.size(0)
    accuracy = correct/total
    print(f"Epoch {epoch+1} : test {accuracy:.4f}")

