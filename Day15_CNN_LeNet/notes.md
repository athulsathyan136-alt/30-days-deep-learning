Day 15: Convolutional Neural Networks (CNNs)

## 🎯 Learning Objectives
- Why MLPs fail on images
- How convolution works
- What a filter/kernel is
- Build LeNet-5 (the first CNN)
- Train on MNIST with 99%+ accuracy

## 🧠 The Problem

MLP on images:
- Each pixel treated independently
- No concept of "nearby"
- Can't detect spatial patterns (edges, shapes)
- Huge parameter count

## 🛠️ The Solution: Convolution

A **filter** (small matrix) slides over the image:
output[i, j] = sum(input[i:i+k, j:j+k] * filter)

text

- Filter size: 3×3 or 5×5 typically
- **Weight sharing**: same filter everywhere
- Captures **local patterns**

## 📊 Key Operations

### 1. Conv2d
```python
nn.Conv2d(in_channels, out_channels, kernel_size)
in_channels: input channels (1 for grayscale, 3 for RGB)

out_channels: number of filters (feature maps to output)

kernel_size: filter size (e.g., 3, 5)

Output shape: (batch, out_channels, H_out, W_out)
Where H_out = H - kernel_size + 1 (no padding)

2. MaxPool2d
python
nn.MaxPool2d(2)
Takes max over 2×2 blocks. Halves H and W.

3. Flatten
Converts (batch, C, H, W) → (batch, C*H*W) before Linear layers.

🏗️ LeNet-5 Architecture
text
Input:  (1, 28, 28)
  ↓ Conv2d(1, 6, 5)  → (6, 24, 24)
  ↓ MaxPool2d(2)     → (6, 12, 12)
  ↓ Conv2d(6, 16, 5) → (16, 8, 8)
  ↓ MaxPool2d(2)     → (16, 4, 4)
  ↓ Flatten          → 256
  ↓ Linear(256, 120) → 120
  ↓ Linear(120, 84)  → 84
  ↓ Linear(84, 10)   → 10
Parameters: ~44,000 (vs 109,386 for MLP — fewer but smarter!)

📊 CNN vs MLP
Aspect	MLP	CNN
Params	More	Fewer
Speed	Slow	Fast
Accuracy	~97%	~99%
Understands space	❌	✅
Translation invariant	❌	✅
💡 Key Takeaways
CNNs are the standard for images

Convolution detects local patterns

Pooling reduces spatial size

Weight sharing = fewer parameters

LeNet-5 (1998) started the CNN era