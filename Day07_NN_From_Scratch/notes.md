# Day 7: Neural Network from Scratch (NumPy only)

## 🎯 Learning Objectives
- Build forward pass with pure NumPy
- Derive backpropagation by hand
- Implement gradient descent manually
- Train a 2-layer NN on MNIST
- Understand what PyTorch does under the hood

## 🧮 The Math of a 2-Layer Network

### Forward Pass
z1 = X @ W1 + b1 # (batch, 784) @ (784, 128) → (batch, 128)
a1 = ReLU(z1) # element-wise
z2 = a1 @ W2 + b2 # (batch, 128) @ (128, 10) → (batch, 10)
probs = softmax(z2) # (batch, 10)

text

### Loss — Cross-Entropy
loss = -mean(log(probs[range(N), y_true]))

text

### Backward Pass (Chain Rule)

Starting from loss:
dZ2 = probs - one_hot(y) # (batch, 10)
dW2 = a1.T @ dZ2 / N # (128, 10)
db2 = dZ2.mean(axis=0) # (10,)

dA1 = dZ2 @ W2.T # (batch, 128)
dZ1 = dA1 * (z1 > 0) # ReLU gradient
dW1 = X.T @ dZ1 / N # (784, 128)
db1 = dZ1.mean(axis=0) # (128,)

text

### Update
W2 -= lr * dW2
b2 -= lr * db2
W1 -= lr * dW1
b1 -= lr * db1

text

## 🔑 Key Derivations

### Softmax + Cross-Entropy Gradient

The beautiful result:
∂loss/∂z2 = probs - one_hot(y)

text

This is why we use softmax + CE together — the gradient is trivially clean.

### ReLU Gradient
ReLU(z) = max(0, z)
ReLU'(z) = (z > 0) # 1 if z>0, else 0

text

## 💡 Why This Matters

PyTorch does all this automatically. But knowing how it works:
- Makes you debug faster
- Lets you write custom layers
- Prevents "I don't know why my model isn't learning"

## 🔗 Next: Day 8 — Activation Functions