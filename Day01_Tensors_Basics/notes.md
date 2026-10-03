# Day 1: Tensors & NumPy Foundations

## 🎯 Learning Objectives
- Understand what a tensor is (0D → ND)
- Master NumPy array operations
- Learn broadcasting rules
- Implement matrix multiplication manually
- Prepare for PyTorch/TensorFlow tensors

## 📚 Key Concepts

### What is a Tensor?
A tensor is a multi-dimensional array — a generalization of scalars, vectors, and matrices.

| Rank | Name    | Example              |
|------|---------|----------------------|
| 0    | Scalar  | `5`                  |
| 1    | Vector  | `[1, 2, 3]`          |
| 2    | Matrix  | `[[1,2],[3,4]]`      |
| 3    | Tensor  | RGB Image (H,W,C)    |
| 4    | Tensor  | Batch of images      |

### Broadcasting Rules
1. Align shapes from the **right**
2. Dimensions must be **equal** or **one of them is 1**
3. Missing dimensions are treated as 1

### Matrix Multiplication
`(m, n) @ (n, p) → (m, p)`  
Inner dimensions must match!

## 💡 Key Takeaways
- Shape errors are the #1 DL bug source
- `reshape`, `transpose`, `broadcasting` are daily tools
- Vectorize everything — avoid Python loops
- Every NN layer = matmul + bias + activation

## 🔗 Next: Day 2 - PyTorch Tensors & GPU