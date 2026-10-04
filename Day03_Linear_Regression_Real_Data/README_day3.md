# Day 3: Linear Regression on Real Data

## ✅ Completed
- [x] Loaded California Housing dataset
- [x] Train/val split (80/20)
- [x] Feature normalization (StandardScaler)
- [x] Multi-feature linear regression (8 → 1)
- [x] Adam optimizer
- [x] Tracked train + val loss
- [x] Visualized predictions & residuals

## 📊 Results
- Train loss: ~0.52
- Val loss: ~0.51
- No overfitting (train ≈ val)
- All 7 exercises passing

## 💭 Reflection
- Normalization is essential for real data
- Adam converges much faster than SGD
- Train/val split reveals overfitting
- Residual plot should be centered on 0

## 🔗 Resources
- California Housing: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html