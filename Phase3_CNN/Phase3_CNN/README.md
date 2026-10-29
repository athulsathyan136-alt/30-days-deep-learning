# Phase 3: CNN on MNIST

## ✅ Completed
- [x] Built LeNet-5 style CNN
- [x] Trained for 10 epochs
- [x] Achieved ~98.6% test accuracy
- [x] Fewer parameters than MLP (44K vs 101K)

## 🏗️ Architecture
Input: (1, 28, 28)
↓ Conv2d(1, 6, 5) + ReLU → (6, 24, 24)
↓ MaxPool2d(2) → (6, 12, 12)
↓ Conv2d(6, 16, 5) + ReLU → (16, 8, 8)
↓ MaxPool2d(2) → (16, 4, 4)
↓ Flatten → (256,)
↓ Linear(256, 128) + ReLU
↓ Linear(128, 64) + ReLU
↓ Linear(64, 10)

**Total parameters: ~44,000**

## 📊 Results

| Epoch | Test Accuracy |
|-------|---------------|
| 1 | 97.97% |
| 5 | 98.77% |
| 7 | **98.89%** (best) |
| 10 | 98.62% |

**Final: ~98.6% accuracy.**

## 🆚 Compare: MLP vs CNN

| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| MLP (Day 6) | 97.66% | 101,770 |
| **CNN (Day 15)** | **98.62%** | **~44,000** |

**CNN: +1% accuracy, 60% fewer parameters.**

## 💭 Reflection
- Convolutions exploit spatial structure
- Weight sharing → fewer parameters
- Max pooling → translation invariance
- Training loop unchanged from `y = 2x + 1`

## 🔗 Next
- Fine-tune: lower LR (0.001), LR scheduler, more epochs
- Push to 99%+