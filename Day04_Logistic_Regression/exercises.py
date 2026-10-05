"""
Day 4 Exercises: Logistic Regression & Binary Classification
Run: python exercises.py
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ex1_sigmoid():
    """Implement sigmoid from scratch."""
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    assert np.isclose(sigmoid(0), 0.5)
    assert sigmoid(100) > 0.99
    assert sigmoid(-100) < 0.01
    assert 0 < sigmoid(5) < 1
    print(f"✅ ex1 passed — σ(0)={sigmoid(0)}, σ(2)={sigmoid(2):.4f}")


def ex2_sigmoid_torch():
    """Same with PyTorch."""
    z = torch.tensor([-2.0, 0.0, 2.0])
    s = torch.sigmoid(z)

    assert s.shape == (3,)
    assert torch.allclose(s, torch.tensor([0.1192, 0.5, 0.8808]), atol=1e-3)
    print(f"✅ ex2 passed — {s.numpy().round(4)}")


def ex3_load_data():
    """Load breast cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert set(y) == {0, 1}
    print(f"✅ ex3 passed — X: {X.shape}, y: {y.shape}, classes: {set(y)}")


def ex4_preprocess():
    """Split + normalize + to tensors."""
    data = load_breast_cancer()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    assert X_train.shape == (455, 30)
    assert y_train.shape == (455, 1)
    print(f"✅ ex4 passed — Train: {X_train.shape}, Val: {X_val.shape}")


def ex5_model_forward():
    """A logistic regression model."""
    model = nn.Linear(30, 1)
    x = torch.randn(8, 30)

    logits = model(x)
    probs = torch.sigmoid(logits)

    assert logits.shape == (8, 1)
    assert probs.shape == (8, 1)
    assert (probs >= 0).all() and (probs <= 1).all()
    print(f"✅ ex5 passed — logits: {logits.shape}, probs: {probs.shape}")


def ex6_bce_loss():
    """Verify BCE loss."""
    criterion = nn.BCEWithLogitsLoss()

    logits = torch.tensor([[10.0], [-10.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = criterion(logits, targets)
    assert loss.item() < 0.01
    print(f"✅ ex6 passed — perfect loss: {loss.item():.6f}")

    logits_wrong = torch.tensor([[-10.0], [10.0]])
    loss_wrong = criterion(logits_wrong, targets)
    assert loss_wrong.item() > 5.0
    print(f"   wrong loss: {loss_wrong.item():.4f}")


def ex7_train_classifier():
    """Full training loop for classification."""
    data = load_breast_cancer()
    X_tr, X_val, y_tr, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(30, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_logits = model(X_val)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()
        acc = (val_preds == y_val).float().mean().item()

    assert acc > 0.9, f"Accuracy too low: {acc}"
    print(f"✅ ex7 passed — Val accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 4 Exercises: Logistic Regression")
    print("=" * 50 + "\n")

    ex1_sigmoid()
    ex2_sigmoid_torch()
    ex3_load_data()
    ex4_preprocess()
    ex5_model_forward()
    ex6_bce_loss()
    ex7_train_classifier()

    print("\n🎉 All Day 4 exercises complete!")