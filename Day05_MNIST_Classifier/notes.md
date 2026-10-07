markdown
# Day 5: Multi-Class Classification (Softmax + MNIST)

## 🎯 Learning Objectives
- Softmax activation
- CrossEntropyLoss
- MNIST dataset (28×28 digits)
- DataLoader batching
- Confusion matrix

## 🔑 Binary vs Multi-Class

| Task | Output | Loss |
|------|--------|------|
| Binary | 1 + sigmoid | BCEWithLogitsLoss |
| Multi-class (K) | K + softmax | CrossEntropyLoss |

## 📊 Softmax
softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

text

Input: K logits. Output: K probabilities summing to 1.
logits = [2.0, 1.0, 0.1]
softmax = [0.659, 0.242, 0.099]

text

## 📉 CrossEntropyLoss
CE = -log( P(true class) )

text

- Takes **logits** (not probabilities)
- Takes **integer targets** (not one-hot)
- Combines softmax + NLL in one stable op

## 🔄 DataLoader

```python
loader = DataLoader(ds, batch_size=64, shuffle=True)
for X, y in loader:
    logits = model(X)
    loss = criterion(logits, y)