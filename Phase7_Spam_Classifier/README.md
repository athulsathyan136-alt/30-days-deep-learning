# 📧 Spam Email Classifier

Classic ML classifier for spam detection using TF-IDF + Logistic Regression.

## 📊 Model Details

- **Algorithm:** TF-IDF + Logistic Regression
- **Dataset:** SMS Spam Collection (5,572 messages)
- **Features:** 5,000 TF-IDF features
- **Test accuracy:** ~97-98%
- **Training time:** ~1-2 seconds

## 🏗️ How It Works
Text → TF-IDF Vector → Logistic Regression → Spam/Ham

text

1. **TF-IDF**: Converts text to numbers based on word importance
2. **Logistic Regression**: Simple linear classifier
3. **Prediction**: Probability of spam

## 🚀 Run It

```bash
pip install scikit-learn pandas
python main.py

Dataset loaded: 5572 messages
Classes: {'ham': 4825, 'spam': 747}

Spam: 747 | Ham: 4825
Train: 4457 | Test: 1115

Vectorizing text with TF-IDF...
Vectorized shape: (4457, 5000)

Training Logistic Regression...

=== Results ===
Test Accuracy: 0.9767

Classification Report:
              precision    recall  f1-score   support

         Ham       0.98      0.99      0.99       966
        Spam       0.96      0.86      0.90       149