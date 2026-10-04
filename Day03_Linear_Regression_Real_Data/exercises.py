
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ex1_load_data():
    """Load California housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target

    assert X.shape == (20640, 8), f"X shape: {X.shape}"
    assert y.shape == (20640,), f"y shape: {y.shape}"
    print(f"✅ ex1 passed — X: {X.shape}, y: {y.shape}")


def ex2_tensor_conversion():
    """Convert NumPy → PyTorch."""
    data = fetch_california_housing()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

    assert X.dtype == torch.float32
    assert y.shape == (20640, 1)
    assert X.shape[1] == 8
    print(f"✅ ex2 passed — X: {X.shape}, y: {y.shape}")


def ex3_train_val_split():
    """Split into train/val."""
    data = fetch_california_housing()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    assert X_train.shape == (16512, 8)
    assert X_val.shape == (4128, 8)
    print(f"✅ ex3 passed — Train: {X_train.shape}, Val: {X_val.shape}")


def ex4_normalization():
    """Standardize features (mean 0, std 1)."""
    data = fetch_california_housing()
    X_train, X_val = train_test_split(data.data, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # After scaling, mean should be ~0, std ~1
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-6)

    # IMPORTANT: val uses TRAIN's statistics (not its own!)
    print(f"✅ ex4 passed — mean: {X_train_scaled.mean():.2e}, std: {X_train_scaled.std():.4f}")


def ex5_model_definition():
    """Define a multi-feature linear model."""
    model = nn.Linear(8, 1)

    assert model.weight.shape == (1, 8)
    assert model.bias.shape == (1,)

    x = torch.randn(32, 8)
    y = model(x)
    assert y.shape == (32, 1)
    print(f"✅ ex5 passed — 8 features → 1 output")


def ex6_train_step():
    """One training step on real data."""
    data = fetch_california_housing()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(8, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = None
    for step in range(100):
        pred = model(X)
        loss = criterion(pred, y)

        if step == 0:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, f"Loss didn't drop enough: {initial_loss:.4f} → {final_loss:.4f}"
    print(f"✅ ex6 passed — loss: {initial_loss:.4f} → {final_loss:.4f}")


def ex7_overfit_check():
    """Verify validation loss is close to train loss."""
    data = fetch_california_housing()
    X_tr, X_val, y_tr, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_tr, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(8, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        pred = model(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        val_loss = criterion(model(X_val_t), y_val_t).item()

    assert val_loss < train_loss * 1.5, f"Overfitting: train={train_loss:.4f}, val={val_loss:.4f}"
    print(f"✅ ex7 passed — Train: {train_loss:.4f}, Val: {val_loss:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 3 Exercises")
    print("=" * 50 + "\n")

    ex1_load_data()
    ex2_tensor_conversion()
    ex3_train_val_split()
    ex4_normalization()
    ex5_model_definition()
    ex6_train_step()
    ex7_overfit_check()

    print("\n🎉 All Day 3 exercises complete!")