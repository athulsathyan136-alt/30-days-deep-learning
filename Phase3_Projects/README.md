# Phase 3: Real Projects

Hands-on projects using real datasets. Same training loop, different data.

## Projects

### 1. Fashion-MNIST Classifier (`fashion_mnist.py`)

**Task:** Classify 10 clothing items from 28×28 grayscale images.

**Dataset:** Fashion-MNIST (60,000 train, 10,000 test)

**Model:** MLP (784 → 128 → 10)

**Expected accuracy:** ~89-92%

**Classes:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**What's different from MNIST:**
- Harder dataset (clothing is more similar than digits)
- Lower accuracy expected (~89% vs 98%)
- Per-class accuracy reveals weak classes (T-shirt vs Shirt is hard)

**How to run:**
```bash
python fashion_mnist.py