import numpy as np

def ex1_relu():
    z = np.array([-2.0,-1.0,0.0,1.0,2.0])
    a = np.maximum(0,z)

    assert np.array_equal(a,[0, 0, 0, 1, 2])
    print(f"✅ ex1 passed — ReLU({z.tolist()}) = {a.tolist()}")

def ex2_relu_grad():
    z = np.array([-2.0,-1.0,0.0,1.0,2.0])
    grad = (z>0).astype(float)

    assert np.array_equal(grad,[0, 0, 0, 1, 1])
    print(f"✅ ex2 passed — ReLU'(z) = {grad.tolist()}")

def ex3_softmax():
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e/e.sum(axis=1,keepdims=True)

    z = np.array([[2.0, 1.0, 0.1]])
    p = softmax(z)

    assert np.isclose(p.sum(),1.0)
    assert np.all(p>0)
    print(f"✅ ex3 passed — softmax: {p.round(4)}")


def ex4_cross_entropy():
    probs = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.2,0.3,0.5]])
    y_true = np.array([0,1,2])

    N = len(y_true)

    correct_probs = probs[np.arange(N),y_true]
    loss = -np.log(correct_probs).mean()

    expected = -(np.log(0.9)+np.log(0.8)+np.log(0.5)) /3
    assert np.isclose(loss,expected)
    print(f"✅ ex4 passed — CE loss: {loss:.4f}")


def ex5_one_hot():
    y = np.array([0,2,1])
    n_class = 4
    one_hot = np.zeros((len(y),n_class))
    one_hot[np.arange(len(y)),y] = 1
    assert one_hot.shape == (3,4)
    assert one_hot[0,0] == 1 and one_hot[1, 2] == 1 and one_hot[2, 1] == 1
    print(f"✅ ex5 passed — one_hot[0] = {one_hot[0].tolist()}")


def ex6_softmax_ce_grad():
    logits = np.array([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.2]])
    y = np.array([0, 1])

    e = np.exp(logits - logits.max(axis=1,keepdims=True))
    probs = e / e.sum(axis=1,keepdims=True)

    grad = probs.copy()
    grad[np.arange(len(y)),y] -= 1

    def ce(logits_, y_):
        e_ = np.exp(logits_ - logits_.max(axis=1, keepdims=True))
        p_ = e_ / e_.sum(axis=1, keepdims=True)
        return -np.log(p_[np.arange(len(y_)), y_]).mean()    

    loss_before = ce(logits,y)
    loss_after = ce(logits-0.01 * grad,y)
    assert loss_after< loss_before
    print(f"✅ ex6 passed — loss went {loss_before:.4f} → {loss_after:.4f}")

def ex7_forward_pass():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4,784))
    w1 = rng.standard_normal((784,128)) * 0.1
    b1 = np.zeros(128)
    w2 =rng.standard_normal((128,10)) * 0.01
    b2 = np.zeros(10)

    z1 = X @ w1 + b1
    a1 = np.maximum(0,z1)
    z2 = a1 @ w2 + b2

    assert z1.shape == (4, 128)
    assert a1.shape == (4, 128)
    assert z2.shape == (4, 10)
    print(f"✅ ex7 passed — forward shapes: X {X.shape} → z2 {z2.shape}")


def ex8_backward_pass():
    rng = np.random.default_rng(0)
    N, D, H, K = 8, 784, 64, 10

    X = rng.standard_normal((N, D))
    y = rng.integers(0, K, N)
    W1 = rng.standard_normal((D, H)) * 0.01
    b1 = np.zeros(H)
    W2 = rng.standard_normal((H, K)) * 0.01
    b2 = np.zeros(K)

    # Forward
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    # Backward
    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1
    dZ2 /= N

    dW2 = a1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    assert dW1.shape == W1.shape
    assert db1.shape == b1.shape
    assert dW2.shape == W2.shape
    assert db2.shape == b2.shape

    def loss_fn(X, y, W1, b1, W2, b2):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        e = np.exp(z2 - z2.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y]).mean()

    lr = 0.1
    loss_before = loss_fn(X, y, W1, b1, W2, b2)
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    loss_after = loss_fn(X, y, W1, b1, W2, b2)

    assert loss_after < loss_before
    print(f"✅ ex8 passed — loss went {loss_before:.4f} → {loss_after:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 7 Exercises: NN from Scratch")
    print("=" * 50 + "\n")

    ex1_relu()
    ex2_relu_grad()
    ex3_softmax()
    ex4_cross_entropy()
    ex5_one_hot()
    ex6_softmax_ce_grad()
    ex7_forward_pass()
    ex8_backward_pass()

    print("\n🎉 All Day 7 exercises complete!")
