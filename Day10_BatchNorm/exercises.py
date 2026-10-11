
import torch
import torch.nn as nn


def ex1_bn_shapes():
    """BatchNorm preserves shape."""
    bn = nn.BatchNorm1d(128)
    x = torch.randn(32, 128)
    y = bn(x)
    assert y.shape == (32, 128)
    print(f"✅ ex1 passed — BatchNorm1d preserves shape {tuple(y.shape)}")


def ex2_bn_normalizes():
    """BatchNorm outputs have ~zero mean, ~unit variance."""
    bn = nn.BatchNorm1d(10)
    bn.train()
    x = torch.randn(64, 10) * 5 + 3     # mean 3, std 5
    y = bn(x)

    assert abs(y.mean().item()) < 1e-5
    assert abs(y.std().item() - 1.0) < 1e-2
    print(f"✅ ex2 passed — output mean: {y.mean():.4f}, std: {y.std():.4f}")


def ex3_bn_learnable_params():
    """BatchNorm has γ and β."""
    bn = nn.BatchNorm1d(10)
    params = list(bn.parameters())
    assert len(params) == 2   # weight (γ), bias (β)
    assert params[0].shape == (10,)
    assert params[1].shape == (10,)
    print(f"✅ ex3 passed — BN has {len(params)} learnable params (γ, β)")


def ex4_bn_train_vs_eval():
    """BN uses batch stats in train, running stats in eval."""
    bn = nn.BatchNorm1d(10)
    bn.train()

    # Feed several batches to update running stats
    for _ in range(50):
        x = torch.randn(32, 10) * 5 + 3
        bn(x)

    # Eval mode: uses running stats
    bn.eval()
    x_test = torch.randn(8, 10) * 5 + 3
    y_eval = bn(x_test)

    # Output should use running_mean ~3, running_std ~5
    # So (x - 3) / 5 should be ~standard normal
    assert abs(y_eval.mean().item()) < 1.0
    print(f"✅ ex4 passed — eval mode uses running stats")


def ex5_bn_placement():
    """Correct placement: Linear → BN → ReLU."""
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    x = torch.randn(32, 784)
    y = model(x)
    assert y.shape == (32, 10)
    print(f"✅ ex5 passed — model with BN works")


def ex6_bn_vs_no_bn():
    """Compare gradient stability with vs without BN."""
    torch.manual_seed(42)

    # Without BN
    layers_no_bn = []
    for _ in range(10):
        layers_no_bn.append(nn.Linear(32, 32))
        layers_no_bn.append(nn.ReLU())
    model_no_bn = nn.Sequential(*layers_no_bn)

    x1 = torch.randn(64, 32, requires_grad=True)
    y1 = model_no_bn(x1).sum()
    y1.backward()
    grad_no_bn = x1.grad.abs().mean().item()

    # With BN
    torch.manual_seed(42)
    layers_bn = []
    for _ in range(10):
        layers_bn.append(nn.Linear(32, 32))
        layers_bn.append(nn.BatchNorm1d(32))
        layers_bn.append(nn.ReLU())
    model_bn = nn.Sequential(*layers_bn)

    x2 = torch.randn(64, 32, requires_grad=True)
    y2 = model_bn(x2).sum()
    y2.backward()
    grad_bn = x2.grad.abs().mean().item()

    print(f"✅ ex6 passed — grad without BN: {grad_no_bn:.6f}")
    print(f"              grad with BN:    {grad_bn:.6f}")


def ex7_bn2d():
    """BatchNorm2d for images: (batch, channels, H, W)."""
    bn = nn.BatchNorm2d(3)   # 3 channels (RGB)
    x = torch.randn(4, 3, 32, 32)
    y = bn(x)
    assert y.shape == (4, 3, 32, 32)
    print(f"✅ ex7 passed — BatchNorm2d preserves shape {tuple(y.shape)}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 10 Exercises: Batch Normalization")
    print("=" * 50 + "\n")

    ex1_bn_shapes()
    ex2_bn_normalizes()
    ex3_bn_learnable_params()
    ex4_bn_train_vs_eval()
    ex5_bn_placement()
    ex6_bn_vs_no_bn()
    ex7_bn2d()

    print("\n🎉 All Day 10 exercises complete!")