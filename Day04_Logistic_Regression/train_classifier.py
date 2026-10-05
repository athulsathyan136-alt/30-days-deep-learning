"""
Day 4: Breast Cancer Classifier
Predicts malignant (0) vs benign (1) from 30 features.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============ 1. Load Data ============
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: Breast Cancer Wisconsin")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)} (0=malignant, 1=benign)")
print(f"Class counts: {np.bincount(y)}")


# ============ 2. Split ============
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")


# ============ 3. Normalize ============
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)


# ============ 4. Model ============
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


model = LogisticRegression(30)


# ============ 5. Loss + Optimizer ============
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ============ 6. Training ============
epochs = 200
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    model.train()
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        train_preds = (torch.sigmoid(logits) > 0.5).float()
        train_acc = (train_preds == y_train).float().mean().item()

        model.eval()
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val).item()
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (val_preds == y_val).float().mean().item()

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# ============ 7. Results ============
print(f"\nFinal Val Accuracy: {val_accs[-1]:.4f}")
print(f"Final Val Loss:     {val_losses[-1]:.4f}")


# ============ 8. Confusion Matrix ============
with torch.no_grad():
    val_probs = torch.sigmoid(model(X_val))
    val_preds = (val_probs > 0.5).float()

y_true = y_val.numpy().flatten()
y_pred = val_preds.numpy().flatten()

tp = ((y_true == 1) & (y_pred == 1)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()

print(f"\nConfusion Matrix (Val):")
print(f"  TP: {tp}  FP: {fp}")
print(f"  FN: {fn}  TN: {tn}")

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ============ 9. Visualize ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label="Train")
axes[0].plot(val_losses, label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_accs, label="Train")
axes[1].plot(val_accs, label="Val")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curve")
axes[1].legend()
axes[1].grid(True)

axes[2].hist(val_probs.numpy()[y_true == 0], bins=20, alpha=0.7, label="Class 0 (malignant)")
axes[2].hist(val_probs.numpy()[y_true == 1], bins=20, alpha=0.7, label="Class 1 (benign)")
axes[2].axvline(0.5, color='r', linestyle='--', label="Threshold")
axes[2].set_xlabel("Predicted Probability")
axes[2].set_ylabel("Count")
axes[2].set_title("Prediction Distribution")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("classifier_results.png", dpi=100)
plt.show()

print("\n✅ Done! Plot saved as classifier_results.png")