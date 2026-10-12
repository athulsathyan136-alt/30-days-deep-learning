# Day 10: Batch Normalization

## 🎯 Learning Objectives
- Understand why deep networks train slowly
- Learn what BatchNorm does
- Apply BatchNorm in PyTorch
- See training speedup vs no BatchNorm

## 🔍 The Problem

Deep networks suffer from:
- Internal covariate shift (layer inputs keep changing)
- Exploding/vanishing gradients
- Slow convergence

## 🛠️ The Solution — BatchNorm

For each mini-batch:
μ = mean(x_batch)
σ = std(x_batch)
x_norm = (x - μ) / (σ + ε)
y = γ * x_norm + β

text

- γ (gamma) — learnable scale
- β (beta) — learnable shift
- ε — tiny number for numerical stability

## 📊 Why It Works

1. Normalizes activations → stable distribution
2. Reduces internal covariate shift → faster learning
3. Allows higher LR → faster convergence
4. Slight regularization effect

## 🧩 In PyTorch

```python
nn.BatchNorm1d(num_features)   # for (batch, features)
nn.BatchNorm2d(num_channels)   # for (batch, C, H, W)
Placement:

text
Linear → BatchNorm → ReLU
🔄 Train vs Eval
Training: uses batch statistics

Evaluation: uses running statistics

💡 Key Takeaways
BN normalizes hidden outputs

Allows higher learning rate

Speeds up convergence 2-3x

Adds learnable params (γ, β)

Different behavior in train vs eval