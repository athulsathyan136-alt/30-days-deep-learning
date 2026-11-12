# 🖼️ Image Captioning with ResNet + LSTM

Encoder-Decoder model that generates captions for images.

## 📊 Architecture

**Encoder:** ResNet-18 (pretrained, frozen) extracts features
**Decoder:** LSTM generates caption word-by-word

## 🎯 What It Does

Given an image, generates a natural language description.

**Example:**
- Input: [dog photo]
- Output: "a dog running through the grass"

## 📁 Files

| File | Purpose |
|------|---------|
| `dataset.py` | Load images + captions |
| `model.py` | ResNet + LSTM architecture |
| `train.py` | Training loop |
| `generate.py` | Generate captions for new images |
| `requirements.txt` | Dependencies |
