
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============ 1. Load Data ============
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

print(f"Features: {data.feature_names}")
print(f"X shape:  {X.shape}")
print(f"y shape:  {y.shape}")
print(f"y range:  [{y.min():.2f}, {y.max():.2f}]")


# ============ 2. Train/Val Split ============
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")


# ============ 3. Normalize ============
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on train
X_val = scaler.transform(X_val)           # ONLY transform val

# To tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)


# ============ 4. Model ============
model = nn.Linear(8, 1)   # 8 features → 1 output
print(f"\nModel: {model}")


# ============ 5. Loss & Optimizer ============
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ============ 6. Training Loop ============
epochs = 200
train_losses = []
val_losses = []

for epoch in range(epochs):
    # --- Train ---
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- Validate ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")


# ============ 7. Results ============
print(f"\nFinal train loss: {train_losses[-1]:.4f}")
print(f"Final val loss:   {val_losses[-1]:.4f}")
print(f"\nLearned weights: {model.weight.data.numpy()}")
print(f"Learned bias:    {model.bias.item():.4f}")


# ============ 8. Visualize ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curves
axes[0].plot(train_losses, label="Train")
axes[0].plot(val_losses, label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True)

# Predictions vs True (val set)
with torch.no_grad():
    preds = model(X_val).numpy().flatten()
trues = y_val.numpy().flatten()

axes[1].scatter(trues, preds, alpha=0.3, s=5)
axes[1].plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--', lw=2)
axes[1].set_xlabel("True Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title("Predicted vs True")
axes[1].grid(True)

# Residuals
residuals = preds - trues
axes[2].hist(residuals, bins=50, alpha=0.7)
axes[2].axvline(0, color='r', linestyle='--')
axes[2].set_xlabel("Residual (Pred - True)")
axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("housing_results.png", dpi=100)
plt.show()

print("\n✅ Done! Plot saved as housing_results.png")