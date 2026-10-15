# Day 11: Weight Initialization

## 🎯 Learning Objectives
- Why random init matters
- Xavier vs He initialization
- What happens with bad init
- When to use which

## 🔍 The Problem

Random init too small → activations shrink to 0 → vanishing gradients
Random init too large → activations explode → NaN loss

## 🛠️ The Solutions

### Xavier / Glorot (for Sigmoid, Tanh)
std = sqrt(1 / fan_in)

text
Where fan_in = number of inputs to the layer.

### He / Kaiming (for ReLU, GELU)
std = sqrt(2 / fan_in)

text
Double the variance because ReLU zeros half the neurons.

### Zero init
❌ Never — all neurons learn the same thing → dead network.

## 🧩 In PyTorch

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
Or use built-in defaults (PyTorch uses Kaiming by default).

📊 What to Compare
Zero init vs random vs He

Too small vs too large

Xavier vs He on ReLU network

💡 Key Takeaways
Init matters — bad init = dead network

He for ReLU, Xavier for Sigmoid/Tanh

PyTorch defaults to Kaiming (good for ReLU)

Always check activation magnitudes if training fails