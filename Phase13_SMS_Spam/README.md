# 📱 SMS Spam Detection with Naive Bayes

Classic text classification comparing Naive Bayes vs Logistic Regression on SMS messages.

## 📊 Model Details

- **Dataset:** SMS Spam Collection (5,572 messages)
- **Algorithm:** Multinomial Naive Bayes + Logistic Regression
- **Features:** 3,000 Count / TF-IDF features
- **Test accuracy:** ~97-98%
- **Training time:** < 1 second

## 🎯 What It Does

Classifies SMS messages as **spam** or **ham** (legitimate).

## 📁 Files

- `main.py` — Complete pipeline with comparison
- `requirements.txt` — Dependencies
- `README.md` — This file

## 🚀 Run It

```bash
pip install scikit-learn pandas
python main.py

Dataset: 5572 messages
Class distribution:
ham     4825
spam     747

Spam: 747 | Ham: 4825

Train: 4457 | Test: 1115

======================================================================
Vectorizer      Classifier           Accuracy
======================================================================
Count (BoW)     Naive Bayes          0.9776
Count (BoW)     Logistic Regression  0.9767
TF-IDF          Naive Bayes          0.9668
TF-IDF          Logistic Regression  0.9749
======================================================================

🏆 Best: Count (BoW) + Naive Bayes (Accuracy: 0.9776)