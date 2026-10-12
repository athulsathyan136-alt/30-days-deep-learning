# Day 10: Batch Normalization

## ✅ Completed
- [x] Understood what BatchNorm does
- [x] Applied BatchNorm1d and BatchNorm2d
- [x] Saw train vs eval behavior
- [x] Trained with vs without BN

## 📊 What BN Does

For each mini-batch:
μ = mean(x)
σ = std(x)
x_norm = (x - μ) / (σ + ε)
y = γ * x_norm + β

text

## 🎯 Why It Works
1. Stable distribution for each layer
2. Reduces internal covariate shift
3. Allows higher learning rate
4. Slight regularization

## 🧩 PyTorch

```python
nn.Sequential(
    nn.Linear(784, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
Placement: Linear → BN → ReLU

🔄 Train vs Eval
Train: uses batch statistics

Eval: uses running statistics