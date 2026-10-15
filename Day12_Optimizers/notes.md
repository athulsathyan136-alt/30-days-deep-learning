# Day 12: Optimizers Deep Dive

## 🎯 Learning Objectives
- Understand what optimizers do
- Compare SGD, Momentum, RMSprop, Adam, AdamW
- Learn when to use each

## 🧠 What an Optimizer Does

After gradients are computed by backprop, the optimizer updates weights:
weight = weight - learning_rate * gradient

text

That's the base (SGD). Better optimizers use smarter update rules.

## 📊 The Optimizers

### 1. SGD (Stochastic Gradient Descent)
w = w - lr * grad

text
- Simple baseline
- Needs careful LR tuning
- Slow but sometimes best final result

### 2. SGD + Momentum
v = β * v_prev + grad
w = w - lr * v

text
- Remembers previous direction
- β typically 0.9
- Smoother, faster than plain SGD

### 3. RMSprop
s = β * s_prev + (1-β) * grad²
w = w - lr * grad / (sqrt(s) + ε)

text
- Per-parameter adaptive LR
- Divides by running RMS of gradients
- Good for RNNs

### 4. Adam
m = β₁ * m_prev + (1-β₁) * grad
s = β₂ * s_prev + (1-β₂) * grad²
m_hat = m / (1 - β₁^t)
s_hat = s / (1 - β₂^t)
w = w - lr * m_hat / (sqrt(s_hat) + ε)

text
- Momentum + RMSprop combined
- Default for most DL
- β₁=0.9, β₂=0.999, ε=1e-8

### 5. AdamW
- Adam + proper weight decay
- Standard for Transformers
- `weight_decay` applied separately

## 🎯 Which to Use?

| Task | Optimizer |
|------|-----------|
| Default | Adam |
| Transformers | AdamW |
| CNNs (older) | SGD + Momentum |
| RNNs | RMSprop |
| Fine-tuning | Adam with small LR |

## 📊 Typical LR

| Optimizer | LR |
|-----------|-----|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam | 1e-4 - 1e-3 |
| AdamW | 1e-4 - 1e-3 |

## 💡 Key Takeaways
- Adam is the default for most DL
- AdamW for Transformers
- SGD + Momentum for image classification (old school)
- Learning rate matters more than optimizer choice

## 🔗 Next: Day 13 — Learning Rate Schedules
