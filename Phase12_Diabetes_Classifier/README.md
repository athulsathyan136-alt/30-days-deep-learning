# 🩺 Diabetes Prediction Classifier

Classic ML classifier comparing 3 algorithms on diabetes data.

## 📊 Model Details

- **Dataset:** Pima Indians Diabetes (442 samples, 10 features)
- **Task:** Binary classification (diabetes progress)
- **Algorithms:** Logistic Regression, Random Forest, Gradient Boosting
- **Test accuracy:** ~75-80%

## 🎯 What It Does

Given 10 medical measurements, predicts if the patient's diabetes will progress.

**Features:**
- Age, Sex, BMI, Blood Pressure
- 6 blood serum measurements (s1-s6)

**Classes:**
- 0: Low diabetes progression
- 1: High diabetes progression

## 📁 Files

- `main.py` — Complete pipeline with 3 classifier comparison
- `requirements.txt` — Dependencies
- `README.md` — This file

## 🚀 Run It

```bash
pip install scikit-learn
python main.py

Dataset loaded: (442, 10)
Features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Classes: [221 221] → 442 samples

Train: 353 | Test: 89

=== Logistic Regression ===
  Accuracy: 0.7416
  AUC:      0.8012

=== Random Forest ===
  Accuracy: 0.7416
  AUC:      0.7891

=== Gradient Boosting ===
  Accuracy: 0.7640
  AUC:      0.8123

Best model: Gradient Boosting (Accuracy: 0.7640, AUC: 0.8123)