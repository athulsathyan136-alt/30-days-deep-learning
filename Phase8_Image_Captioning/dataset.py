
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import urllib.request
import zipfile


# Use a small curated sample (100 images with captions) for demo
SAMPLE_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
CAPTIONS_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"


def download_data():
    """Download Flickr8k (small sample)."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    images_zip = "data/flickr8k_images.zip"
    captions_zip = "data/flickr8k_text.zip"
    
    # Note: Full dataset is large (~1 GB). For demo, use small subset.
    # In production, you'd download from Kaggle:
    # https://www.kaggle.com/datasets/adityajn105/flickr8k
    
    print("For this project, download Flickr8k from Kaggle:")
    print("https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print("Extract to: data/images/ and data/captions.txt")
    print()
    print("Or use your own image-caption pairs.")


class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab=None, transform=None, max_len=20):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.max_len = max_len
        
        # Load captions
        self.captions = {}
        with open(captions_file, "r") as f:
            for line in f.readlines()[1:]:  # Skip header
                img_id, caption = line.strip().split(",", 1)
                if img_id not in self.captions:
                    self.captions[img_id] = []
                self.captions[img_id].append(caption)
        
        self.image_ids = list(self.captions.keys())
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.idx_to_word = {i: w for w, i in self.vocab.items()}
    
    def _build_vocab(self):
        vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        for captions in self.captions.values():
            for caption in captions:
                for word in caption.lower().split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
        return vocab
    
    def _tokenize(self, caption):
        tokens = [self.vocab["<start>"]]
        for word in caption.lower().split():
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))
        tokens.append(self.vocab["<end>"])
        
        # Pad or truncate
        if len(tokens) < self.max_len:
            tokens += [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len - 1] + [self.vocab["<end>"]]
        
        return torch.tensor(tokens)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Use first caption
        caption = self.captions[img_id][0]
        tokens = self._tokenize(caption)
        
        return image, tokens


if __name__ == "__main__":
    print("Dataset module ready.")
    print("Download Flickr8k from:")
    print("https://www.kaggle.com/datasets/adityajn105/flickr8k")