# Day 9: Regularization — Fighting Overfitting

## 🎯 Learning Objectives
- Understand overfitting vs underfitting
- Apply dropout, weight decay, early stopping
- Detect overfitting from loss curves
- Build models that generalize

## 🧠 The Core Problem

| | Train Acc | Val Acc | Diagnosis |
|--|-----------|---------|-----------|
| Underfitting | Low | Low | Model too simple |
| Good fit | High | High | ✅ Perfect |
| Overfitting | Very High | Low | Model memorized training data |

## 🔍 Detecting Overfitting

Good:
Train Loss: 1.0 → 0.5 → 0.3 → 0.25
Val Loss: 1.0 → 0.5 → 0.35 → 0.30

text

Overfitting:
Train Loss: 1.0 → 0.5 → 0.2 → 0.05 ← keeps dropping
Val Loss: 1.0 → 0.5 → 0.6 → 0.85 ← goes UP

text

## 🛡️ Regularization Techniques

### 1. Dropout
Randomly zero neurons during training.
```python
nn.Dropout(p=0.5)
Only active during model.train()

Disabled at model.eval()

2. Weight Decay (L2)
python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
3. Data Augmentation
python
transforms.RandomHorizontalFlip()
transforms.RandomCrop(28, padding=4)
4. Early Stopping
Stop when val loss starts rising.

💡 Key Takeaways
Overfitting = memorizing training data

Dropout = random neuron removal

Weight decay = penalize large weights

Always evaluate on validation set