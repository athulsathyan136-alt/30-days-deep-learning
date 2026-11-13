# 🔢 Handwritten Digit Recognition

CNN classifier with interactive drawing interface. Draw a digit, model predicts which one.

## 📊 Model Details

- **Architecture:** Small CNN (2 conv + 2 linear layers)
- **Dataset:** MNIST (60,000 train + 10,000 test)
- **Training:** 5 epochs
- **Test accuracy:** ~98-99%
- **Parameters:** ~430,000

## 🏗️ Architecture
nput: (1, 28, 28) grayscale
↓ Conv2d(1, 32, 3) + ReLU + MaxPool(2) → (32, 14, 14)
↓ Conv2d(32, 64, 3) + ReLU + MaxPool(2) → (64, 7, 7)
↓ Flatten → (3136,)
↓ Linear(3136, 128) + ReLU
↓ Linear(128, 10)
Output: 10 digit probabilities
