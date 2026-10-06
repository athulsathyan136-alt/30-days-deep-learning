# Day 5: Multi-Class Classification (MNIST)

## ✅ Completed
- [x] Softmax (manual + PyTorch)
- [x] CrossEntropyLoss
- [x] MNIST dataset loading
- [x] DataLoader batching
- [x] Softmax regression classifier
- [x] Per-class accuracy + confusion matrix
- [x] Saved model

## 📊 Results
- Test accuracy: ~92%
- Confusion matrix shows which digits are confused (often 4↔9, 3↔5)

## 💭 Reflection
- Softmax + CE = multi-class standard
- CrossEntropyLoss takes logits + integer targets
- DataLoader handles batching
- Adding hidden layers will boost accuracy to ~98% (Day 6)

## 🔗 Resources
- MNIST: http://yann.lecun.com/exdb/mnist/
- CrossEntropyLoss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html