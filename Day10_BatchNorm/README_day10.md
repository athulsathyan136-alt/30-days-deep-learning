# Day 10: Batch Normalization

## ✅ Completed
- [x] Understood why deep networks train slowly
- [x] Applied BatchNorm1d and BatchNorm2d
- [x] Saw train vs eval mode behavior
- [x] Compared with vs without BN

## 📊 What BN Does

For each mini-batch:


- γ (scale) and β (shift) are learnable
- Normalizes activations to mean=0, std=1

## 🎯 Why It Works

1. Stable input distribution for each layer
2. Reduces internal covariate shift
3. Allows higher learning rate
4. Slight regularization effect

## 🧩 PyTorch Usage

```python
nn.Sequential(
    nn.Linear(784, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10),
)