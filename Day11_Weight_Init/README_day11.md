# Day 11: Weight Initialization

## ✅ Completed
- [x] Zero init — dead network
- [x] Too small — vanishing
- [x] Too large — exploding
- [x] Xavier init (for sigmoid/tanh)
- [x] He init (for ReLU)
- [x] Applied to real MNIST model
- [x] Compared 5 inits side-by-side

## 📊 Init Formulas

| Init | std | Use for |
|------|-----|---------|
| Zero | 0 | ❌ Never |
| Xavier | sqrt(1 / fan_in) | Sigmoid, Tanh |
| He | sqrt(2 / fan_in) | ReLU, GELU, SiLU |

## 🧩 PyTorch

```python
def init_fn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_fn)
🔑 Key Insights
Bad init → dead network (zero, huge)

Good init → healthy activations

He for ReLU, Xavier for Sigmoid

PyTorch's defaults are He-like (good for ReLU)

💭 Reflection
Initialization is one of the most underrated DL tricks

A single line can change training from broken to working

PyTorch defaults to Kaiming/He, which is why most code "just works"