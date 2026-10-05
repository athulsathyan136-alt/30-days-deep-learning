# Day 3: Linear Regression on Real Data

## 🎯 Learning Objectives
- Load and explore a real dataset
- Proper train/val/test split
- Feature normalization (why + how)
- Multi-feature linear regression
- Adam optimizer
- Detect overfitting

## 📊 Dataset: California Housing

- Source: 1990 California census
- 20,640 samples
- 8 features:
  - MedInc (median income)
  - HouseAge
  - AveRooms
  - AveBedrms
  - Population
  - AveOccup
  - Latitude
  - Longitude
- Target: Median house value (in $100k)

## 🔑 Key Concepts

### Why Split Data?

| Split | Purpose | Typical % |
|-------|---------|-----------|
| Train | Learn parameters | 70-80% |
| Validation | Tune hyperparameters | 10-15% |
| Test | Final evaluation | 10-15% |

**Never evaluate on training data** — model memorizes it.

### Why Normalize?

Real features have wildly different scales:
- MedInc: 0.5 – 15
- Population: 3 – 35,000

**Without normalization:**
- Large features dominate
- Gradients explode/vanish
- Training is slow or unstable

**With normalization** (z-score):