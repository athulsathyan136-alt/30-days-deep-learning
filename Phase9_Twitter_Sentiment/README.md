# 🐦 Twitter Sentiment Analysis

Multi-class sentiment classifier for tweets using classic ML.

## 📊 Model Details

- **Task:** 3-class classification (negative, neutral, positive)
- **Dataset:** Sentiment140 (1.6M tweets, using 10,000 subset)
- **Features:** 10,000 TF-IDF features (unigrams + bigrams)
- **Models:** Logistic Regression vs Naive Bayes
- **Expected accuracy:** ~65-70%

## 🎯 Why This Is Harder Than IMDB

| IMDB | Twitter |
|------|---------|
| 2 classes | 3 classes |
| Long reviews | Short tweets (280 chars) |
| Clean text | Hashtags, emojis, slang |
| 50K examples | 1.6M examples |
| ~92% accuracy | ~70% accuracy |

**Twitter sentiment is genuinely harder because:**
- Sarcasm: "Great, another Monday 🙄" (negative)
- Slang: "This is sick!" (positive, not illness)
- Short texts: less context
- Noisy: hashtags, mentions, URLs

## 📁 Files

- `main.py` — Complete pipeline
- `requirements.txt` — Dependencies
- `README.md` — This file

## 🚀 Run It

```bash
pip install scikit-learn pandas numpy
python main.py