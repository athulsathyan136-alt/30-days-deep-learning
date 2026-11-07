from fastapi import FastAPI          
from pydantic import BaseModel        
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Load model
import torch                        

app = FastAPI(title="BERT Sentiment API")

MODEL_NAME = "Athul-sathyan/bert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class TextInput(BaseModel):
    text:str

@app.get("/health")
def health():
    return {
        "status":"ok"
    }

@app.post("/predict")
def predict(input:TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length = 128
    )

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits

    prob = torch.softmax(logits,dim=1)
    predicted_class = logits.argmax(1).item()
    confidence = prob[0,predicted_class].item()

    sentiment = "positive" if predicted_class == 1 else "negative"

    return{
        "text":input.text,
        "sentiment":sentiment,
        "confidence":round(confidence,4)
    }




