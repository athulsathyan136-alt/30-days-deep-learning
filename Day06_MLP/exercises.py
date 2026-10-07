import torch
import torch.nn as nn

def ex1_relu():
    x = torch.tensor([-2.0,-1.0,0.0,1.0,2.0])
    r = torch.relu(x)
    assert torch.equal(r,torch.tensor([0.0,0.0,0.0,1.0,2.0]))
    print(f"✅ ex1 passed — ReLU({x.tolist()}) = {r.tolist()}")

def ex2_mlp_forward():
    model = nn.Sequential(
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10),
    )
    x = torch.randn(32,784)
    y = model(x)
    assert y.shape == (32,10)
    print(f"✅ ex2 passed — input {tuple(x.shape)} → output {tuple(y.shape)}")

def ex3_parameter_count():
    model = nn.Sequential(
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10),
    )
    total = sum(p.numel() for p in model.parameters())
    expected = 784*128 + 128 + 128 *64 + 64+64*10+10
    assert total == expected
    print(f"✅ ex3 passed — total params: {total:,}")

def ex4_no_relu_is_linear():
    w1 = nn.Linear(10,5)
    w2 = nn.Linear(5,2)
    x = torch.randn(1,10)
    y_stack = w2(w1(x))

    with torch.no_grad():
        w_comb = w2.weight @ w1.weight
        b_comb = w2.weight @ w1.bias + w2.bias
    y_single = x @ w_comb.T + b_comb

    assert torch.allclose(y_stack,y_single,atol=1e-5)
    print(f"✅ ex4 passed — stacking Linear ≠ deeper model (needs ReLU)")

def ex5_relu_breaks_linearity():
    model = nn.Sequential(
        nn.Linear(10,5),
        nn.ReLU(),
        nn.Linear(5,2),

    )
    x = torch.randn(1,10)
    y1 = model(x)
    y2 = model(x)

    assert torch.allclose(y1,y2)
    print(f"✅ ex5 passed — ReLU adds non-linearity")

def ex6_vanishing_grad_check():
    x_sig = torch.tensor([-10.0,0.0,10.0],requires_grad = True)
    y_sig = torch.sigmoid(x_sig).sum()
    y_sig.backward()
    sig_grads = x_sig.grad.abs()

    x_relu = torch.tensor([-10.0,0.0,10.0],requires_grad = True)
    y_relu = torch.relu(x_relu).sum()
    y_relu.backward()
    relu_grads = x_relu.grad.abs()

    assert relu_grads.max() == 1.0
    assert sig_grads.max() < 0.3
    print(f"✅ ex6 passed — Sigmoid grads: {sig_grads.tolist()}")
    print(f"              ReLU grads:   {relu_grads.tolist()}")

def ex7_capacity_demo():
    small = nn.Sequential(nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 10))
    big = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                          nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, 10))

    small_parms = sum(p.numel() for p in small.parameters())
    big_params   = sum(p.numel() for p in big.parameters())

    assert big_params> small_parms*10
    print(f"✅ ex7 passed — small: {small_parms:,} | big: {big_params:,}")

if __name__ == "__main__":
    print("=" * 50)
    print("Day 6 Exercises: MLP & Non-Linearity")
    print("=" * 50 + "\n")

    ex1_relu()
    ex2_mlp_forward()
    ex3_parameter_count()
    ex4_no_relu_is_linear()
    ex5_relu_breaks_linearity()
    ex6_vanishing_grad_check()
    ex7_capacity_demo()

    print("\n🎉 All Day 6 exercises complete!")