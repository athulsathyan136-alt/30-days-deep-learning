# Day 12: Optimizers Deep Dive

## ✅ Completed
- [x] SGD — baseline
- [x] SGD + Momentum
- [x] RMSprop
- [x] Adam
- [x] AdamW
- [x] Compared 5 optimizers on MNIST

## 📊 Optimizer Comparison

| Optimizer | Idea | Best For |
|-----------|------|----------|
| SGD | w -= lr * grad | Simple, baseline |
| SGD + Momentum | Add velocity | CNNs (old) |
| RMSprop | Adaptive per-param LR | RNNs |
| Adam | Momentum + RMSprop | Default |
| AdamW | Adam + weight decay | Transformers |

## 🧩 PyTorch

```python
# SGD
torch.optim.SGD(params, lr=0.01)

# SGD + Momentum
torch.optim.SGD(params, lr=0.01, momentum=0.9)

# RMSprop
torch.optim.RMSprop(params, lr=0.001)

# Adam
torch.optim.Adam(params, lr=0.001)

# AdamW
torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
🔑 Key Insights
Adam is the default for most DL

AdamW is standard for transformers

SGD + Momentum sometimes wins on CNNs

LR matters more than optimizer choice

💭 Reflection
Same model, different optimizers → different convergence speeds

Adam converges fastest but needs lower LR

Optimizer choice is about ~1-2% accuracy