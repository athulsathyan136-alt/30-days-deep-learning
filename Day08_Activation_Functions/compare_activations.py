import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.relu(x)

def leaky_relu(x):
    return F.leaky_relu(x,negative_slope=0.1)

def gelu(x):
    return F.gelu(x)

def silu(x):
    return F.silu(x)

activations = {
    "Sigmoid":   sigmoid,
    "Tanh":      tanh,
    "ReLU":      relu,
    "LeakyReLU": leaky_relu,
    "GELU":      gelu,
    "SiLU":      silu,
}

x = torch.linspace(-5,5,500,requires_grad=True)

fig,axes = plt.subplots(2,6,figsize=(20,6))

for i, (name,fn) in enumerate(activations.items()):
    x_i = x.clone().detach().requires_grad_(True)
    y = fn(x_i)

    axes[0,i].plot(x_i.detach().numpy(),y.detach().numpy(),lw=2)
    axes[0,i].set_title(name)
    axes[0,i].grid(True)
    axes[0,i].axhline(0,color='k',lw=0.5)
    axes[0,i].axvline(0,color='k',lw=0.5)

    y.sum().backward()
    grad = x_i.grad.detach().numpy()

    axes[1, i].plot(x_i.detach().numpy(), grad, lw=2, color='crimson')
    axes[1, i].set_title(f"{name} gradient")
    axes[1, i].grid(True)
    axes[1, i].axhline(0, color='k', lw=0.5)
    axes[1, i].axvline(0, color='k', lw=0.5)

axes[0,0].set_ylabel('Output')
axes[1,0].set_ylabel('Gradient')


plt.tight_layout()
plt.savefig('activation_comparison.png',dpi=100)
plt.show()

print("✅ Saved activations_comparison.png")

print("\n" + "=" * 70)
print(f"{'Activation':<12} {'f(-1)':<10} {'f(0)':<10} {'f(1)':<10} {'grad(-1)':<10} {'grad(0)':<10} {'grad(1)':<10}")
print("=" * 70)
for name,fn in activations.items():
    x_i = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = fn(x_i)
    vals = y.detach().tolist()
    y.sum().backward()
    grads = x_i.grad.detach().tolist()

    print(f"{name:<12} {vals[0]:<10.4f} {vals[1]:<10.4f} {vals[2]:<10.4f} "
          f"{grads[0]:<10.4f} {grads[1]:<10.4f} {grads[2]:<10.4f}")

print('='*70)

print("\n📊 Vanishing Gradient Demo")
print("-" * 70)
print("Build a 20-layer network with each activation.")
print("Check the gradient magnitude at the input.\n")

for name , fn in activations.items():
    torch.manual_seed(42)
    layers = []
    for _ in range(20):
        layers.append(nn.Linear(32,32))
        layers.append({'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh, 'ReLU': nn.ReLU, 'LeakyReLU': lambda: nn.LeakyReLU(0.1), 'GELU': nn.GELU, 'SiLU': nn.SiLU}[name]())
    model = nn.Sequential(*layers)

    x_in = torch.randn(4,32,requires_grad=True)
    out = model(x_in).sum()
    out.backward()
    grad_mag = x_in.grad.abs().mean().item()

    bar = "█" * int(min(50, -np.log10(grad_mag + 1e-10)))
    print(f"{name:<12} grad_norm = {grad_mag:.8f}  {bar}")