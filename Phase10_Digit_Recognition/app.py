"""
Gradio app for MNIST digit recognition.
Draw a digit, model predicts which one.
"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ============================================================
# 1. Model class (same as training)
# ============================================================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 2. Load trained model
# ============================================================
model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn.pth", map_location="cpu"))
model.eval()


# ============================================================
# 3. Preprocess sketch input
# ============================================================
def preprocess_sketch(sketch):
    """
    Convert Gradio sketch input to (1, 1, 28, 28) tensor.
    Gradio sketch gives us a (280, 280, 4) RGBA image.
    """
    if sketch is None:
        return None
    
    # Extract alpha channel (the drawn strokes)
    if sketch.shape[-1] == 4:
        img = sketch[:, :, 3]  # alpha
    else:
        img = sketch[:, :, 0]  # grayscale fallback
    
    # Resize to 28×28
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to tensor, normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    tensor = (tensor - 0.1307) / 0.3081  # MNIST normalization
    
    return tensor


# ============================================================
# 4. Predict function
# ============================================================
def predict(sketch):
    """Predict digit from sketch."""
    if sketch is None:
        return {"Please draw a digit": 0.0}
    
    x = preprocess_sketch(sketch)
    if x is None:
        return {"Error": 0.0}
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Return dict: {digit_label: probability}
    return {str(i): float(probs[i]) for i in range(10)}


# ============================================================
# 5. Gradio UI
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        type="numpy",
        label="Draw a digit (0-9)",
        image_mode="RGBA",
        brush=gr.Brush(colors=["#000000"], color_mode="fixed")
    ),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🔢 Digit Recognition",
    description="Draw a digit (0-9) with your mouse or finger. The model will predict which digit it is.",
    examples=None,
    live=False
)

if __name__ == "__main__":
    demo.launch()