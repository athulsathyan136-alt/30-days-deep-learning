# Day 8: Activation Functions Deep Dive

## 🎯 Learning Objectives
- Compare all major activation functions
- Understand vanishing gradients
- Learn when to use each
- Visualize forward and gradient behavior

## 📊 The Activations

### 1. Sigmoid
σ(x) = 1 / (1 + e^(-x))
Range: (0, 1)
Gradient max: 0.25

text

**Pros:** Outputs probability, smooth
**Cons:** Vanishing gradient (saturates), not zero-centered
**Use:** Binary classification output layer

### 2. Tanh
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
Range: (-1, 1)
Gradient max: 1.0

text

**Pros:** Zero-centered (better than sigmoid)
**Cons:** Still saturates
**Use:** RNN hidden states (historically)

### 3. ReLU
ReLU(x) = max(0, x)
Range: [0, ∞)
Gradient: 0 or 1

text

**Pros:** Fast, no saturation for positive, sparse
**Cons:** Dead neurons (gradient 0 for x<0 forever)
**Use:** CNNs, MLPs — modern default

### 4. LeakyReLU
LeakyReLU(x) = max(0.01·x, x)
Range: (-∞, ∞)

text

**Pros:** Fixes dead neurons
**Cons:** Extra hyperparameter (slope)
**Use:** GANs (discriminator), when ReLU dies

### 5. GELU
GELU(x) = x · Φ(x) (Φ = standard normal CDF)
Range: (-0.17, ∞)

text

**Pros:** Smooth, better gradients
**Cons:** Slower to compute
**Use:** BERT, GPT, ViT — modern transformers

### 6. SiLU / Swish
SiLU(x) = x · σ(x)
Range: (-0.28, ∞)

text

**Pros:** Smooth, self-gated
**Cons:** Slower
**Use:** EfficientNet, YOLOv5

## 🧊 Vanishing Gradient Problem

For sigmoid:
σ'(x) = σ(x)·(1-σ(x))
Max = 0.25 (at x=0)

text

For a 10-layer net with sigmoid:
Gradient after 10 layers ≈ 0.25^10 ≈ 9.5e-7

text
→ Gradient vanishes → deep layers don't learn.

ReLU fixes this:
ReLU'(x) = 1 for x>0
Gradient after 10 layers = 1 (if all positive)

text

## 📊 Quick Reference

| Activation | Range | Vanishing? | Used in |
|------------|-------|------------|---------|
| Sigmoid | (0, 1) | ✅ Yes | Binary output |
| Tanh | (-1, 1) | ✅ Yes | Old RNNs |
| ReLU | [0, ∞) | ❌ No | CNNs, MLPs |
| LeakyReLU | (-∞, ∞) | ❌ No | GANs |
| GELU | (-0.17, ∞) | ❌ No | Transformers |
| SiLU | (-0.28, ∞) | ❌ No | EfficientNet, YOLO |

## 💡 Rules of Thumb

1. **Default:** ReLU
2. **Transformers:** GELU
3. **GAN discriminator:** LeakyReLU
4. **Output layer:**
   - Binary: sigmoid
   - Multi-class: softmax (not really an activation — it's a normalization)
   - Regression: none (linear)
5. **Never use sigmoid/tanh in hidden layers of deep nets** (vanishing gradient)