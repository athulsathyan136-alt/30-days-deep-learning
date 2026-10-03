import torch


def ex1_tensor_creation():
    scalar = torch.tensor(42.0)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    matrix = torch.zeros(2, 3)
    tensor = torch.ones(3, 4, 5)

    assert scalar.shape == torch.Size([])
    assert vector.shape == torch.Size([4])
    assert matrix.shape == torch.Size([2, 3])
    assert tensor.shape == torch.Size([3, 4, 5])
    print("✅ ex1 passed")


def ex2_tensor_ops():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])

    add = a + b
    mul = a * b
    dot = torch.dot(a, b)

    assert torch.equal(add, torch.tensor([11.0, 22.0, 33.0, 44.0]))
    assert torch.equal(mul, torch.tensor([10.0, 40.0, 90.0, 160.0]))
    assert dot.item() == 300.0
    print("✅ ex2 passed")


def ex3_autograd_basic():
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y.backward()

    # dy/dx = 2x + 2 = 8
    assert x.grad is not None
    assert abs(x.grad.item() - 8.0) < 1e-6
    print("✅ ex3 passed")


def ex4_autograd_chain():
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3
    z = y * 2
    w = z + 5
    w.backward()

    # dw/dx = 1 * 2 * 3x² = 24
    assert abs(x.grad.item() - 24.0) < 1e-6
    print("✅ ex4 passed")


def ex5_manual_gradient_descent():
    """Fit y = 3x with manual gradient descent."""
    X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = torch.tensor([3.0, 6.0, 9.0, 12.0, 15.0])

    w = torch.tensor(0.0, requires_grad=True)
    lr = 0.01

    for step in range(1000):
        y_pred = w * X
        loss = ((y_pred - Y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()

    assert abs(w.item() - 3.0) < 0.1, f"w = {w.item()}"
    print(f"✅ ex5 passed — learned w = {w.item():.4f} (expected 3.0)")


def ex6_training_loop():
    """Full training loop with optimizer. y = 2x + 3"""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    Y = torch.tensor([[5.0], [7.0], [9.0], [11.0], [13.0]])

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(2000):
        y_pred = model(X)
        loss = criterion(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w = model.weight.item()
    b = model.bias.item()
    assert abs(w - 2.0) < 0.1, f"w = {w}"
    assert abs(b - 3.0) < 0.1, f"b = {b}"
    print(f"✅ ex6 passed — learned w = {w:.4f}, b = {b:.4f} (expected 2.0, 3.0)")


def ex7_gpu_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    assert x.device.type == device.type
    print(f"✅ ex7 passed — using device: {device}")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}\n")
    ex1_tensor_creation()
    ex2_tensor_ops()
    ex3_autograd_basic()
    ex4_autograd_chain()
    ex5_manual_gradient_descent()
    ex6_training_loop()
    ex7_gpu_check()
    print("\n🎉 All Day 2 exercises complete!")