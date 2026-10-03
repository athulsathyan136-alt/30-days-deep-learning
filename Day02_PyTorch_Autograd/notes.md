# Day 2: PyTorch Tensors & Autograd

## 🎯 Learning Objectives
- PyTorch tensors vs NumPy arrays
- Autograd — automatic differentiation
- The core training loop
- Train a real model on real data

## 🔑 PyTorch vs NumPy

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| GPU | ❌ | ✅ |
| Autograd | ❌ | ✅ |
| Syntax | `np.array` | `torch.tensor` |
| Gradients | manual | automatic |

## 🔥 Autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)   # 8.0