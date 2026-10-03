# Day 2: PyTorch Tensors & Autograd

## ✅ Completed
- [x] PyTorch tensor creation & operations
- [x] Autograd — basic + chain rule
- [x] Manual gradient descent
- [x] Full training loop
- [x] Trained a linear regressor

## 📊 Results
- All 7 exercises passing
- Model learned w ≈ 2.0, b ≈ 3.0

## 💭 Reflection
The training loop is universal:
forward → loss → zero_grad → backward → step

Every DL model uses this exact pattern.

## 🔗 Resources
- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html