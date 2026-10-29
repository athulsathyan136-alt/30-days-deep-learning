# Day 15: CNNs — LeNet-5

## ✅ Completed
- [x] Understood why MLPs fail on images
- [x] Learned Conv2d, MaxPool2d, Flatten
- [x] Built LeNet-5 from scratch
- [x] Trained on MNIST (~99% accuracy)
- [x] Visualized feature maps

## 🏗️ LeNet-5 Architecture
nput: (1, 28, 28)
↓ Conv2d(1, 6, 5) → (6, 24, 24)
↓ ReLU
↓ MaxPool2d(2) → (6, 12, 12)
↓ Conv2d(6, 16, 5) → (16, 8, 8)
↓ ReLU
↓ MaxPool2d(2) → (16, 4, 4)
↓ Flatten → 256
↓ Linear(256, 120)
↓ Linear(120, 84)
↓ Linear(84, 10)

text

**Parameters: ~44,000**

## 📊 CNN vs MLP

| Aspect | MLP (Day 6) | CNN (Day 15) |
|--------|-------------|--------------|
| Params | 109K | 44K |
| Test Acc | ~97% | ~99% |
| Speed | Slower | Faster |
| Spatial awareness | ❌ | ✅ |

## 🧩 Core Operations

```python
nn.Conv2d(in_channels, out_channels, kernel_size)
nn.MaxPool2d(2)
nn.Flatten()
💭 Reflection
CNNs are fundamentally different from MLPs

Filters detect local patterns

Weight sharing = fewer params

LeNet-5 is 26 years old but still instructive

Modern CNNs (ResNet, EfficientNet) build on these ideas