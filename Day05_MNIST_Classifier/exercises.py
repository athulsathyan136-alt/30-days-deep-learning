import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ex1_softmax_manual():
    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    z = np.array([2.0, 1.0, 0.1])
    s = softmax(z)

    assert np.isclose(s.sum(), 1.0)
    assert np.all(s > 0)
    assert s[0] > s[1] > s[2]
    print(f"✅ ex1 passed — softmax: {s.round(4)}")


def ex2_softmax_torch():
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    probs = F.softmax(logits, dim=1)

    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs.shape == (1, 3)
    print(f"✅ ex2 passed — {probs.numpy().round(4)}")


def ex3_cross_entropy():
    criterion = nn.CrossEntropyLoss()

    logits = torch.tensor([[0.0, 0.0, 10.0]])
    target = torch.tensor([2])
    loss = criterion(logits, target)
    assert loss.item() < 0.01
    print(f"✅ ex3 passed — perfect loss: {loss.item():.6f}")

    logits_wrong = torch.tensor([[10.0, 0.0, 0.0]])
    loss_wrong = criterion(logits_wrong, target)
    assert loss_wrong.item() > 5.0
    print(f"   wrong loss: {loss_wrong.item():.4f}")


def ex4_cross_entropy_vs_manual():
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    target = torch.tensor([0])

    ce = nn.CrossEntropyLoss()(logits, target)
    probs = F.softmax(logits, dim=1)
    manual = -torch.log(probs[0, 0])

    assert torch.allclose(ce, manual, atol=1e-5)
    print(f"✅ ex4 passed — CE: {ce.item():.4f}, manual: {manual.item():.4f}")


def ex5_batch_targets():
    logits = torch.randn(4, 10)
    targets = torch.tensor([3, 7, 0, 9])

    loss = nn.CrossEntropyLoss()(logits, targets)
    assert loss.item() > 0
    print(f"✅ ex5 passed — batch CE loss: {loss.item():.4f}")


def ex6_predictions():
    logits = torch.tensor([[1.0, 3.0, 0.5],
                           [5.0, 1.0, 2.0],
                           [0.1, 0.2, 0.3]])

    preds = logits.argmax(dim=1)
    assert torch.equal(preds, torch.tensor([1, 0, 2]))
    print(f"✅ ex6 passed — preds: {preds.tolist()}")


def ex7_accuracy():
    logits = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))

    logits[torch.arange(50), targets[:50]] = 100.0

    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()

    assert acc >= 0.5
    print(f"✅ ex7 passed — accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 5 Exercises: Multi-Class Classification")
    print("=" * 50 + "\n")

    ex1_softmax_manual()
    ex2_softmax_torch()
    ex3_cross_entropy()
    ex4_cross_entropy_vs_manual()
    ex5_batch_targets()
    ex6_predictions()
    ex7_accuracy()

    print("\n🎉 All Day 5 exercises complete!")