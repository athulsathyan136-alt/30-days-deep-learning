# Phase 3: CNN Projects

## ✅ Completed

### 1. Basic CNN (`lesson15_cnn.py`)
- LeNet-style: 2 Conv2d + 2 MaxPool + 3 Linear
- **Test accuracy: 98.62%**

### 2. Improved CNN (`lesson15b_improved_cnn.py`)
- Modern architecture: padding + BatchNorm + Dropout
- 2 Conv blocks (32, 64 filters) + classifier
- **Test accuracy: 99.25%**

## 📊 Comparison

| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| MLP (Day 6) | 97.66% | 101,770 |
| Basic CNN | 98.62% | ~44,000 |
| Improved CNN | 99.25% | ~422,000 |

## 🏗️ Improved CNN Architecture
Input: (1, 28, 28)

Block 1:
Conv2d(1, 32, 3, padding=1) → BatchNorm2d(32) → ReLU
MaxPool2d(2) → Dropout(0.25)
→ (32, 14, 14)

Block 2:
Conv2d(32, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
MaxPool2d(2) → Dropout(0.25)
→ (64, 7, 7)

Classifier:
Flatten → (3136,)
Linear(3136, 128) → BatchNorm1d(128) → ReLU → Dropout(0.5)
Linear(128, 10)


**Total parameters: ~422,000**

## 🎯 Techniques Used

| Technique | Purpose | Effect |
|-----------|---------|--------|
| Convolution | Detect local patterns | +1% over MLP |
| Padding=1 | Preserve spatial size | Allows deeper nets |
| BatchNorm | Stabilize training | Faster convergence |
| Dropout | Prevent overfitting | Better generalization |
| MaxPool | Reduce spatial size | Fewer parameters |

## 💭 Key Insights
- BatchNorm makes epoch 1 already 97.85% (vs ~94% without)
- More filters (32, 64) capture more patterns
- Dropout prevents the large model from overfitting
- 99.25% is excellent for this simple CNN

## 🔗 Next Steps
- Fine-tune: LR scheduler, data augmentation
- Push to 99.5%+
- Or: try pretrained ResNet