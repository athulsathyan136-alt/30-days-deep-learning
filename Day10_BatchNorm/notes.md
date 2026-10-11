# Day 10: Batch Normalization

## 🎯 Learning Objectives
- Understand why deep networks train slowly
- Learn what BatchNorm does
- Apply BatchNorm in PyTorch
- See training speedup vs no BatchNorm

## 🔍 The Problem

Deep networks suffer from:
- **Internal covariate shift** — each layer's input distribution keeps changing
- **Exploding/vanishing gradients**
- **Slow convergence**

## 🛠️ The Solution

Batch Normalization (BN):
μ = mean(x_batch)
σ = std(x_batch)
x_norm = (x - μ) / (σ + ε)
y = γ * x_norm + β

text

- `γ` (gamma) — learnable scale
- `β` (beta) — learnable shift
- `ε` — tiny number for numerical stability

## 📊 Why It Works

1. **Normalizes activations** → stable distribution
2. **Reduces internal covariate shift** → faster learning
3. **Allows higher LR** → faster convergence
4. **Slight regularization** → less overfitting

## 🧩 In PyTorch

```python
nn.BatchNorm1d(num_features)   # for (batch, features)
nn.BatchNorm2d(num_features)   # for (batch, channels, H, W)
Placement:

text
Linear → BatchNorm → ReLU
BN before activation is standard.

🔄 Train vs Eval Mode
Training: uses batch statistics

Evaluation: uses running statistics (computed during training)

This is why model.train() / model.eval() matter.

💡 Key Takeaways
BN normalizes hidden layer outputs

Allows higher learning rate

Speeds up convergence 2-3x

Adds learnable parameters (γ, β)

Different behavior in train vs eval