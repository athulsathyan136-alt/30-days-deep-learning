# Day 6: MLP — Multi-Layer Perceptron

## 🎯 Learning Objectives
- Why single-layer models fail
- Hidden layers + non-linearity
- ReLU activation
- MLP architecture
- Boost MNIST accuracy to ~98%

## ❌ The Problem with Single-Layer Models

`y = Wx + b` is LINEAR. Stacking two linear layers = still linear:
Layer1: h = W1·x + b1
Layer2: y = W2·h + b2
= W2·(W1·x + b1) + b2
= (W2·W1)·x + (W2·b1 + b2)
= W_combined·x + b_combined

text
Combining two linear layers = one linear layer. No power gained.

## ✅ The Fix: Non-Linear Activation

Insert a non-linear function between layers:
h = ReLU(W1·x + b1)
y = W2·h + b2

text
Now the whole thing is non-linear — can approximate ANY function (Universal Approximation Theorem).

## 🔥 ReLU
ReLU(x) = max(0, x)

text
- Positive → pass through
- Negative → 0
- Cheap to compute
- Gradient is 0 or 1 — no vanishing gradient

Alternatives: Sigmoid, Tanh, LeakyReLU, GELU.

## 🏗️ MLP Architecture for MNIST
Input: 784 (28×28 pixels)
↓ Linear(784, 128) + ReLU
Hidden: 128
↓ Linear(128, 64) + ReLU
Hidden: 64
↓ Linear(64, 10)
Output: 10 logits

text

Parameters: 784×128+128 + 128×64+64 + 64×10+10 = **~109,000**

## 💡 Key Takeaways
- Depth + non-linearity = power
- Always ReLU between Linear layers
- NEVER put ReLU on the last layer (want raw logits)
- More params → more capacity → risk of overfitting
- No softmax in the model — CrossEntropyLoss handles it

## 🔗 Next: Day 7 — NN from Scratch (NumPy only)