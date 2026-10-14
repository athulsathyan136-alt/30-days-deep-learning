import torch
import torch.nn as nn
import math


def ex1_zero_init():
    model = nn.Linear(10,5)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    x = torch.randn(4,10)
    y = model(x)

    assert torch.allclose(y[:,0],y[:,1])
    print(f"✅ ex1 passed — zero init makes all outputs identical")

def ex2_too_small_init():
    model = nn.Sequential(
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100)
    )
    for m in model:
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=1e-4)
            nn.init.zeros_(m.bias)
    x = torch.randn(32,100)
    y = model(x)
    assert y.abs().mean().item() < 1e-3
    print(f"✅ ex2 passed — tiny init output magnitude: {y.abs().mean():.2e}")

def ex3_too_large_init():
    model = nn.Sequential(
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100)
    )
    for m in model:
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=10.0)
            nn.init.zeros_(m.bias)
    x = torch.randn(32,100)
    y = model(x)
    assert y.abs().mean().item() > 1e3
    print(f"✅ ex3 passed — huge init output magnitude: {y.abs().mean():.2e}")

def ex4_xavier_init():
    layer = nn.Linear(100,50)
    nn.init.xavier_normal_(layer.weight)

    expected_std = math.sqrt(2.0/(100+50))
    actual_std = layer.weight.std().item()
    assert abs(actual_std - expected_std) < 0.01
    print(f"✅ ex4 passed — Xavier std: {actual_std:.4f} (expected {expected_std:.4f})")

def ex5_he_init():
    layer = nn.Linear(100,50)
    nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')

    expected_std = math.sqrt(2.0/100)
    actual_std = layer.weight.std().item()
    assert abs(actual_std-expected_std)<0.01
    print(f"✅ ex5 passed — He std: {actual_std:.4f} (expected {expected_std:.4f})")

def ex6_apply_init():
    def init_fn(m):
        if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            nn.init.zeros_(m.bias)

    model = nn.Sequential(
        nn.Linear(784,256),nn.ReLU(),
        nn.Linear(256,128),nn.ReLU(),
        nn.Linear(128,10),
    )
    model.apply(init_fn)
    for m in model:
        if isinstance(m,nn.Linear):
            assert torch.all(m.bias==0)
    print(f"✅ ex6 passed — applied He init to all Linear layers")

def ex7_activation_scale():
    torch.manual_seed(42)
    x = torch.randn(64,784)

    result = {}
    for name , init_fn in [
      ("zero",  lambda w: nn.init.zeros_(w)),
        ("small", lambda w: nn.init.normal_(w, std=1e-3)),
        ("large", lambda w: nn.init.normal_(w, std=1.0)),
        ("xavier", lambda w: nn.init.xavier_normal_(w)),
        ("he",     lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')),  
    ]:

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10), 
        )

        for m in model:
            if isinstance(m,nn.Linear):
                init_fn(m.weight)
                nn.init.zeros_(m.bias)

        y = model(x)
        result[name] = y.std().item()
    for name, std in result.items():
        print(f"   {name:<8} output std: {std:.4f}")


    print(f"✅ ex7 passed — compared activation scales")

if __name__ == "__main__":
    print("=" * 50)
    print("Day 11 Exercises: Weight Initialization")
    print("=" * 50 + "\n")

    ex1_zero_init()
    ex2_too_small_init()
    ex3_too_large_init()
    ex4_xavier_init()
    ex5_he_init()
    ex6_apply_init()
    ex7_activation_scale()

    print("\n🎉 All Day 11 exercises complete!")