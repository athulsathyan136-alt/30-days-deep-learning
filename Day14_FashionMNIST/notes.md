# Day 14: Fashion-MNIST Project

## 🎯 Learning Objectives
- Build a complete ML pipeline
- Apply all techniques from Days 8-13
- Use data augmentation
- Save/load best model
- Evaluate on real test set

## 📊 Fashion-MNIST

- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- 60K train, 10K test
- 28×28 grayscale
- **Harder than MNIST** (~92% vs ~98%)

## 🏗️ Architecture
Input: 784
↓ Linear(784, 256) + BN + ReLU + Dropout(0.3)
Hidden: 256
↓ Linear(256, 128) + BN + ReLU + Dropout(0.3)
Hidden: 128
↓ Linear(128, 64) + BN + ReLU + Dropout(0.3)
Hidden: 64
↓ Linear(64, 10)
Output: 10 logits

text

## 🎯 Techniques Used

| Technique | Day | Why |
|-----------|-----|-----|
| BatchNorm | 10 | Faster training |
| Dropout | 9 | Prevent overfitting |
| He init | 11 | Better start |
| Adam | 12 | Better optimizer |
| CosineAnnealing | 13 | Smooth LR decay |
| Data augmentation | 9 | More effective data |

## 🧩 Data Augmentation

For Fashion-MNIST:
```python
transforms.RandomHorizontalFlip()    # mirror image
transforms.RandomRotation(10)        # slight rotation
transforms.RandomAffine(0, translate=(0.1, 0.1))
📊 Expected Results
Test accuracy: ~92%

Training time: 5-10 min on CPU

💡 Key Takeaways
Real ML = combine many techniques

Best model = save when val acc improves

Data augmentation helps a lot

Fashion-MNIST is a good "next step" after MNIST