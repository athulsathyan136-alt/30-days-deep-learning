

import torch
import torch.nn as nn


def ex1_dropout_shape():
    x = torch.ones(1000)
    drop = nn.Dropout(p=0.5)
    drop.train()
    y = drop(x)
    assert y.shape == x.shape
    zeros = (y == 0).sum().item()
    assert 400 < zeros < 600
    print(f"✅ ex1 passed — dropout zeroed {zeros}/1000 values")


def ex2_dropout_eval_mode():
    x = torch.ones(1000)
    drop = nn.Dropout(p=0.5)
    drop.eval()
    y = drop(x)
    assert torch.equal(x, y)
    print(f"✅ ex2 passed — dropout is identity in eval mode")


def ex3_weight_decay():
    torch.manual_seed(42)
    model_a = nn.Linear(10, 10)
    model_b = nn.Linear(10, 10)
    model_b.load_state_dict(model_a.state_dict())

    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1, weight_decay=0.1)

    X = torch.randn(32, 10)
    y = torch.randn(32, 10)

    for _ in range(20):
        loss_a = ((model_a(X) - y) ** 2).mean()
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()

        loss_b = ((model_b(X) - y) ** 2).mean()
        opt_b.zero_grad(); loss_b.backward(); opt_b.step()

    norm_a = model_a.weight.norm().item()
    norm_b = model_b.weight.norm().item()

    assert norm_b < norm_a
    print(f"✅ ex3 passed — no decay: {norm_a:.4f}, with decay: {norm_b:.4f}")


def ex4_data_augmentation():
    from torchvision import transforms
    aug = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
    img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    flipped = aug(img)
    assert flipped.shape == img.shape
    assert flipped[0, 0, 0].item() == img[0, 0, 3].item()
    print(f"✅ ex4 passed — augmentation preserves shape")


def ex5_early_stopping_logic():
    val_losses = [1.0, 0.8, 0.6, 0.55, 0.58, 0.62, 0.70, 0.75]
    best = float('inf')
    patience = 2
    patience_counter = 0
    stop_epoch = None

    for epoch, loss in enumerate(val_losses):
        if loss < best:
            best = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_epoch = epoch
                break

    assert stop_epoch == 5
    print(f"✅ ex5 passed — stopped at epoch {stop_epoch}")


def ex6_overfit_detection():
    train_losses = [1.0, 0.5, 0.3, 0.15, 0.08, 0.03]
    val_losses   = [1.0, 0.6, 0.5, 0.55, 0.70, 0.90]

    for i in range(1, len(val_losses)):
        if val_losses[i] > val_losses[i-1]:
            overfit_epoch = i
            break

    assert overfit_epoch == 3
    print(f"✅ ex6 passed — overfitting starts at epoch {overfit_epoch}")


def ex7_regularized_model():
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 10),
    )
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout
    x = torch.randn(32, 784)
    y = model(x)
    assert y.shape == (32, 10)
    print(f"✅ ex7 passed — model with dropout works")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 9 Exercises: Regularization")
    print("=" * 50 + "\n")

    ex1_dropout_shape()
    ex2_dropout_eval_mode()
    ex3_weight_decay()
    ex4_data_augmentation()
    ex5_early_stopping_logic()
    ex6_overfit_detection()
    ex7_regularized_model()

    print("\n🎉 All Day 9 exercises complete!")