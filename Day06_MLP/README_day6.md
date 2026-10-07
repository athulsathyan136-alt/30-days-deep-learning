# Day 6: MLP — Multi-Layer Perceptron

## 🎯 Learning Objectives
- Understand why single-layer models are limited
- Add non-linearity with ReLU activation
- Build a multi-layer neural network
- Boost MNIST accuracy from 92% → 98%

## ✅ Completed
- [x] ReLU activation (max(0, x))
- [x] Proof that stacked Linear layers = one Linear layer
- [x] Proof that ReLU breaks linearity
- [x] Sigmoid vs ReLU gradient comparison
- [x] MLP with 2 hidden layers
- [x] Trained on MNIST with ~98% accuracy

## 🏗️ Architecture
Input: 784 (28×28 pixels)
↓ Linear(784, 128) + ReLU
Hidden: 128
↓ Linear(128, 64) + ReLU
Hidden: 64
↓ Linear(64, 10)
Output: 10 logits (digits 0-9)

text

**Total parameters:** 109,386

## 📊 Results

| Metric | Day 5 (Softmax) | Day 6 (MLP) |
|--------|-----------------|-------------|
| Test Accuracy | ~92% | **~98%** |
| Parameters | 7,850 | 109,386 |
| Layers | 1 | 3 |
| Non-linearity | None | ReLU |

## 🧠 Key Concepts

### Why Non-Linearity Matters

Without activation, stacking layers is pointless:
Layer1: h = W1·x + b1
Layer2: y = W2·h + b2
= W2·(W1·x + b1) + b2
= (W2·W1)·x + (W2·b1 + b2)
= W_combined·x + b_combined

text

Two linear layers = one linear layer. No gain.

### ReLU
ReLU(x) = max(0, x)

text

- Positive → pass through
- Negative → 0
- Gradient: 0 or 1 (no vanishing)
- 99% of modern networks use ReLU or variants

### Why MLP Beats Softmax Regression

**Softmax regression:** each pixel contributes linearly to class scores.

**MLP:** hidden neurons detect non-linear patterns (curves, edges, loops), then output layer combines them. Depth = composition of features.

## 💭 Reflection

- Non-linearity is what makes deep networks powerful
- 14× more parameters gave 6% accuracy gain
- ReLU has ~4× larger max gradient than sigmoid (0.25 vs 1.0 for positive inputs)
- MLP still ignores spatial structure — CNNs (Day 15) will exploit it

## 🔗 Next: Day 7 — Neural Network from Scratch (NumPy only)