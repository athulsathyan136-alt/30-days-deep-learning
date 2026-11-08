"""
BERT Sentiment Classifier - Gradio App
Uses fine-tuned DistilBERT from HuggingFace Hub.
"""
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


MODEL_NAME = "Athul-sathyan/bert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def predict_sentiment(text):
    """Classify text as positive or negative."""
    if not text or not text.strip():
        return "Please enter some text.", 0.0
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = torch.softmax(logits, dim=1)
    predicted_class = logits.argmax(1).item()
    confidence = probs[0, predicted_class].item()
    
    sentiment = "positive" if predicted_class == 1 else "negative"
    return sentiment, round(confidence, 4)


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Movie Review", lines=3),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Number(label="Confidence")
    ],
    title="🎭 BERT Sentiment Classifier",
    description="Fine-tuned DistilBERT on IMDB movie reviews.",
    examples=[
        ["This movie was absolutely amazing!"],
        ["Terrible film, complete waste of time."],
        ["It was okay, nothing special."]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)