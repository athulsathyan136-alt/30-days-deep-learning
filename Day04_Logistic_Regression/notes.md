# Day 4: Logistic Regression — Binary Classification

## 🎯 Learning Objectives
- Classification vs Regression
- Sigmoid activation
- Binary Cross-Entropy loss
- Accuracy, precision, recall, F1
- Train a real classifier on medical data

## 🔑 Regression vs Classification

| Task | Output | Example |
|------|--------|---------|
| Regression | Continuous number | House price |
| Classification | Category | Tumor benign/malignant |

## 📊 Sigmoid Function
σ(z) = 1 / (1 + e^(-z))

- Input: any real number
- Output: between 0 and 1
- Interpretation: probability

Decision rule: `σ(z) > 0.5` → predict class 1.

## 📉 Binary Cross-Entropy
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

- Confident wrong predictions = huge loss
- Use `nn.BCEWithLogitsLoss` (combines sigmoid + BCE, stable)

## 📊 Metrics
Precision = TP / (TP + FP) ← quality of positive predictions
Recall = TP / (TP + FN) ← coverage of actual positives
F1 = 2·P·R / (P + R) ← harmonic mean