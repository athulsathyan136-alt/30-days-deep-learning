# Day 7: Neural Network from Scratch (NumPy)

## ✅ Completed
- [x] ReLU forward + backward
- [x] Softmax + Cross-Entropy
- [x] One-hot encoding
- [x] Full forward pass in NumPy
- [x] Full backward pass (manual chain rule)
- [x] Trained 2-layer NN on MNIST
- [x] ~97% test accuracy — no PyTorch!

## 🧮 The Complete Math

### Forward
z1 = X @ W1 + b1
a1 = ReLU(z1)
z2 = a1 @ W2 + b2
probs = softmax(z2)

text

### Backward
dZ2 = probs - one_hot(y)
dW2 = a1.T @ dZ2 / N
db2 = mean(dZ2)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (z1 > 0)
dW1 = X.T @ dZ1 / N
db1 = mean(dZ1)

text

### Update
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2

text

## 📊 Results
- Test accuracy: ~97%
- No PyTorch. No autograd. Pure NumPy.

## 💭 Reflection
- PyTorch's `.backward()` computes exactly what I wrote by hand
- The softmax + CE gradient is beautifully simple: `probs - one_hot`
- ReLU gradient is either 0 or 1
- Manual backprop makes me appreciate autograd much more

## 🔗 Next: Day 8 — Activation Functions
▶️ Step 2: Run Exercises
powershell
python exercises.py
Expected:

text
==================================================
Day 7 Exercises: NN from Scratch
==================================================

✅ ex1 passed — ReLU([-2.0, -1.0, 0.0, 1.0, 2.0]) = [0, 0, 0, 1, 2]
✅ ex2 passed — ReLU'(z) = [0, 0, 0, 1, 1]
✅ ex3 passed — softmax: [[0.659  0.242  0.099]]
✅ ex4 passed — CE loss: 0.3285
✅ ex5 passed — one_hot[0] = [1.0, 0.0, 0.0, 0.0]
✅ ex6 passed — loss went ...
✅ ex7 passed — forward shapes: X (4, 784) → z2 (4, 10)
✅ ex8 passed — loss went ...

🎉 All Day 7 exercises complete!