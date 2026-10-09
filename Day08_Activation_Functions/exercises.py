import numpy as np
import torch.nn as nn
import torch

def ex1_sigmoid():
    x = torch.tensor([-5.0,0.0,5.0])
    s = torch.sigmoid(x)

    assert torch.allclose(s,torch.tensor([0.0067,0.5,0.9933]),atol=1e-3)
    print(f"✅ ex1 passed — sigmoid({x.tolist()}) = {s.numpy().round(4)}")

def ex2_tanh():
    x = torch.tensor([-5.0,0.0,5.0])
    t = torch.tanh(x)

    assert torch.allclose(t,torch.tensor([-0.9999,0.0,0.9999]),atol=1e-3)
    print(f"✅ ex2 passed — tanh({x.tolist()}) = {t.numpy().round(4)}")

def ex3_relu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    r = torch.relu(x)

    assert torch.equal(r,torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    print(f"✅ ex3 passed — relu: {r.tolist()}")

def ex4_leaky_relu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    lr = torch.nn.functional.leaky_relu(x,negative_slope=0.01)
    assert abs(lr[0].item() - (-0.02)) < 1e-6
    assert abs(lr[-1].item() - 2.0) < 1e-6
    print(f"✅ ex4 passed — leaky_relu: {lr.tolist()}")

def ex5_gelu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    g = torch.nn.functional.gelu(x)

    assert abs(g[2].item()) < 1e-5
    assert abs(g[-1].item() - 2.0) < 0.05
    print(f"✅ ex5 passed — gelu: {g.numpy().round(4)}")


def ex6_vanishing_gradient():
    x = torch.tensor([-10.0, 0.0, 10.0], requires_grad=True)
    y = torch.sigmoid(x).sum()
    y.backward()
    sig_grads = x.grad.abs()

    assert sig_grads.max() <= 0.25 + 1e-6
    assert sig_grads[0] < 1e-3
    assert sig_grads[2] < 1e-3
    print(f"✅ ex6 passed — sigmoid grads: {sig_grads.tolist()}")
    print(f"              max: {sig_grads.max():.4f} (vanishes at extremes)")

def ex7_relu_gradient():
    x = torch.tensor([-10.0, 0.0, 10.0], requires_grad=True)
    y = torch.relu(x).sum()
    y.backward()
    relu_grad = x.grad

    assert relu_grad[0].item() == 0.0
    assert relu_grad[2].item() == 1.0
    print(f"✅ ex7 passed — relu grads: {relu_grad.tolist()}")

def ex8_deep_network_gradient():
    torch.manual_seed(42)

    sigmoid_stack = nn.Sequential(*[nn.Linear(32,32),nn.Sigmoid()] * 10)
    x_sig = torch.randn(4,32,requires_grad=True)
    y_sig = sigmoid_stack(x_sig).sum()
    y_sig.backward()
    sig_first_grad = x_sig.grad.abs().mean().item()

    torch.manual_seed(42)
    relu_stack = nn.Sequential(*[nn.Linear(32, 32), nn.ReLU()] * 10)
    x_relu = torch.randn(4, 32, requires_grad=True)
    y_relu = relu_stack(x_relu).sum()
    y_relu.backward()
    relu_first_grad = x_relu.grad.abs().mean().item()

    assert sig_first_grad < relu_first_grad * 0.5
    print(f"✅ ex8 passed — grad after 20 layers:")
    print(f"   Sigmoid: {sig_first_grad:.6f} (vanished)")
    print(f"   ReLU:    {relu_first_grad:.6f} (healthy)")

if __name__ == "__main__":
    print("=" * 50)
    print("Day 8 Exercises: Activation Functions")
    print("=" * 50 + "\n")

    ex1_sigmoid()
    ex2_tanh()
    ex3_relu()
    ex4_leaky_relu()
    ex5_gelu()
    ex6_vanishing_gradient()
    ex7_relu_gradient()
    ex8_deep_network_gradient()

    print("\n🎉 All Day 8 exercises complete!")

