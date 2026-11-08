# 🎭 BERT Sentiment Classifier

Fine-tuned DistilBERT model for movie review sentiment analysis.

## 🚀 Live Demo

**[Try it on HuggingFace Spaces](https://athul-sathyan-bert-sentiment.hf.space)**

## 📊 Model Details

- **Base model:** DistilBERT (`distilbert-base-uncased`)
- **Dataset:** IMDB (50,000 movie reviews)
- **Training:** 1,000 samples, 1 epoch
- **Test accuracy:** ~82.6%
- **Task:** Binary classification (positive/negative)
- **Model hub:** [Athul-sathyan/bert-sentiment](https://huggingface.co/Athul-sathyan/bert-sentiment)

## 🛠️ Tech Stack

- PyTorch
- HuggingFace Transformers
- Gradio (web UI)
- HuggingFace Datasets

## 🎯 How It Works

1. User enters a movie review
2. Text tokenized using DistilBERT tokenizer
3. Token IDs fed to fine-tuned DistilBERT
4. Model outputs 2 logits (negative, positive)
5. Softmax converts to probabilities
6. Predicted class + confidence returned

## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio web app |
| `train_and_save.py` | Training script |
| `requirements.txt` | Dependencies |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py