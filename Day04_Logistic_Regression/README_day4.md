# Day 4: Logistic Regression

## ✅ Completed
- [x] Sigmoid function (from scratch + PyTorch)
- [x] Breast cancer dataset (30 features)
- [x] Train/val split + normalization
- [x] Logistic regression model
- [x] BCEWithLogitsLoss training
- [x] Confusion matrix + precision/recall/F1

## 📊 Results
- Val accuracy: ~97%
- Precision: ~97%
- Recall: ~98%
- F1: ~98%

## 💭 Reflection
- BCEWithLogitsLoss is more stable than sigmoid+BCE
- Model outputs raw logits, sigmoid only for interpretation
- Accuracy alone is misleading — use precision/recall/F1
- Prediction probability histogram shows clean class separation

## 🔗 Resources
- https://en.wikipedia.org/wiki/Sigmoid_function
- https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html