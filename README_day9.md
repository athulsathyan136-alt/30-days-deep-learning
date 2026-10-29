# Day 9: Overfitting & Regularization

## ✅ Completed
- [x] Dropout
- [x] Weight decay (L2)
- [x] Data augmentation
- [x] Early stopping logic
- [x] Detecting overfitting
- [x] Trained on 500 samples to force overfitting

## 📊 Techniques Summary

| Technique | How | PyTorch |
|-----------|-----|---------|
| Dropout | Random neuron removal | `nn.Dropout(0.5)` |
| Weight decay | Penalize large weights | `optimizer(..., weight_decay=1e-4)` |
| Data aug | More variants | `transforms.RandomHorizontalFlip()` |
| Early stopping | Stop when val rises | Manual patience loop |

## 💭 Reflection
- Overfitting: train loss ↓, val loss ↑
- Dropout only active during training
- Weight decay keeps weights small
- Evaluate on validation, not training

## 🔗 Next: Day 10 — Batch Normalization