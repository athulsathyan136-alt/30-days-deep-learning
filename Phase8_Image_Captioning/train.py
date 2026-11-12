
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from model import ImageCaptioner
import os


def main():
    # Check dataset
    if not os.path.exists("data/captions.txt"):
        print("Dataset not found!")
        print("Download Flickr8k from: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("Extract to: data/images/ and data/captions.txt")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = CaptionDataset("data/images", "data/captions.txt")
    print(f"Dataset size: {len(dataset)} | Vocab: {len(dataset.vocab)}")
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model
    model = ImageCaptioner(vocab_size=len(dataset.vocab)).to(device)
    
    # Loss + optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    # Training
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device)
            
            # Predict all tokens except last
            logits = model(images, captions[:, :-1])  # (batch, seq_len-1, vocab)
            
            # Reshape for loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Save
    torch.save(model.state_dict(), "captioner.pth")
    print("✅ Model saved to captioner.pth")


if __name__ == "__main__":
    main()