import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 200
X = torch.linspace(-5,5,N).unsqueeze(1)
true_w,true_b = 2.0,3.0
Y = true_w*X + true_b + 0.5 * torch.randn(N,1)

print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')

class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearRegressor()
print(f'\nModel: {model}')
print(f'Initial weight : {model.linear.weight.item():.4f}')
print(f'Initial bias : {model.linear.bias.item():.4f}')

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)

epochs = 200
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred,Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1)%20 == 0:
        print(f'Epoch :{epoch+1:3d} | Loss : {loss.item():.6f}')

print(f"\nLearned weight: {model.linear.weight.item():.4f} (true: {true_w})")
print(f"Learned bias:   {model.linear.bias.item():.4f} (true: {true_b})")

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

with torch.no_grad():
    y_fit = model(X)

ax2.scatter(X.numpy(),Y.numpy(),alpha=0.4,label='Data')
ax2.plot(X.numpy(),y_fit.numpy(),'r-',linewidth=2,label='Prediction')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Fitted Line')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_result.png',dpi=100)
plt.show()

print("\n✅ Training complete! Plot saved as training_result.png")