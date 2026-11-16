# 🍷 Wine Quality Classifier

Classic ML classifier using Random Forest on the Wine dataset.

## 📊 Model Details

- **Dataset:** Wine (178 samples, 13 features, 3 classes)
- **Algorithm:** Random Forest (100 trees)
- **Test accuracy:** ~97%
- **Training time:** < 1 second

## 🎯 What It Does

Given 13 chemical measurements of a wine, predicts which **cultivar** (grape variety) it belongs to.

**Classes:**
- Class 0: Cultivar 1
- Class 1: Cultivar 2
- Class 2: Cultivar 3

**Features (13):**
Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315, Proline

## 📁 Files

- `main.py` — Complete pipeline
- `requirements.txt` — Dependencies
- `README.md` — This file

## 🚀 Run It

```bash
pip install scikit-learn
python main.py


Expected output:
Dataset loaded: (178, 13)
Features: ['alcohol', 'malic_acid', ...]
Classes: ['class_0' 'class_1' 'class_2']
Samples per class: [59, 71, 48]

Train: 142 | Test: 36
Features normalized

✅ Random Forest trained (100 trees)

=== Results ===
Test Accuracy: 0.9722

=== Feature Importance ===
  flavanoids: 0.1856
  proline: 0.1623
  color_intensity: 0.1234
  ...