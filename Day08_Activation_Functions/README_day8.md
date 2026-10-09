# Day 8: Activation Functions Deep Dive

## 🎯 Learning Objectives
- Compare all major activation functions
- Understand vanishing gradients
- Learn when to use each activation
- Visualize forward and gradient behavior

## ✅ Completed
- [x] Sigmoid — probability output
- [x] Tanh — zero-centered sigmoid
- [x] ReLU — modern default
- [x] LeakyReLU — fixes dead neurons
- [x] GELU — used in transformers
- [x] SiLU — used in EfficientNet
- [x] Vanishing gradient demonstration
- [x] 20-layer gradient comparison

## 📊 Summary Table

| Activation | Range | Vanishing? | Modern Use |
|------------|-------|-----------|------------|
| Sigmoid | (0, 1) | ✅ Yes | Binary output layer |
| Tanh | (-1, 1) | ✅ Yes | Legacy RNNs |
| ReLU | [0, ∞) | ❌ No | CNNs, MLPs (default) |
| LeakyReLU | (-∞, ∞) | ❌ No | GANs |
| GELU | (-0.17, ∞) | ❌ No | Transformers (BERT, GPT) |
| SiLU | (-0.28, ∞) | ❌ No | EfficientNet, YOLO |

## 🧮 Activation Formulas

### Sigmoid