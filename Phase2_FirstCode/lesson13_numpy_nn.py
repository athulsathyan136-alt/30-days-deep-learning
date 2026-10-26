import numpy as np
#y = 2x+1
x =  np.array([[1],[2],[3],[4],[5]])
y = np.array([[3],[5],[7],[9],[11]])

np.random.seed(42)
w = np.random.randn()*0.01
b  = 0.0
lr = 0.01
for epoch in range(200):
    y_pred = w*x+b
    error = y_pred - y
    loss = np.mean(error**2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)

    w = w - lr * dw
    b = b - lr * db

print(f"Learned weight: {w:.4f}")
print(f"Learned bias: {b:.4f}")
