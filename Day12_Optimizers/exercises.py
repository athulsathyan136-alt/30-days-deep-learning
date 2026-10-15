import torch
import torch.nn as nn

def ex1_sgd_update():
    w = torch.tensor([1.0],requires_grad=True)
    loss = (w**2).sum()
    loss.backward()

    optimizer = torch.optim.SGD([w],lr=0.1)
    optimizer.step()

    assert abs(w.item()- 0.8) < 1e-6
    print(f"✅ ex1 passed — SGD step: w = {w.item():.4f} (expected 0.8)")

def ex2_momentum_exists():
    model = nn.Linear(10,10)
    opt = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    x = torch.randn(4,10)
    y = model(x).sum()
    y.backward()
    opt.step()
    assert len(opt.state) > 0
    for param, state in opt.state.items():
        assert 'momentum_buffer' in state
    print(f"✅ ex2 passed — SGD+Momentum has velocity buffer")

def ex3_rmsprop():
    model = nn.Linear(10,10)
    opt = torch.optim.RMSprop(model.parameters(),lr=0.01)

    x = torch.randn(4,10)
    y = model(x).sum()
    y.backward()
    opt.step()

    for param,state in opt.state.items():
        assert 'square_avg' in state
    print(f"✅ ex3 passed — RMSprop tracks square_avg")

def ex4_adam():
    model = nn.Linear(10,10)
    opt = torch.optim.Adam(model.parameters(),lr=0.001)

    x = torch.randn(4,10)
    y = model(x).sum()
    y.backward()
    opt.step()

    for param,state in opt.state.items():
        assert 'exp_avg' in state
        assert 'exp_avg_sq' in state
    print(f"✅ ex4 passed — Adam has 2 moment buffers")

def ex5_adamw():
    model = nn.Linear(10,10)
    opt = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)

    x = torch.randn(4,10)
    y = model(x).sum()
    y.backward()
    opt.step()

    assert len(opt.state) > 0
    print(f"✅ ex5 passed — AdamW works with weight decay")

def ex6_convergence_speed():

    torch.manual_seed(42)
    X = torch.randn(200,10)
    y = (X.sum(dim=1,keepdim=True)>0).float()

    def make_model():
        torch.manual_seed(42)
        return nn.Sequential(nn.Linear(10,32),nn.ReLU(),nn.Linear(32,1))

    model_sgd = make_model()
    opt_sgd= torch.optim.SGD(model_sgd.parameters(),lr=0.01)
    losses_sgd = []
    for _ in range(100):
        pred = model_sgd(X)
        loss = nn.functional.mse_loss(pred,y)
        opt_sgd.zero_grad();loss.backward();opt_sgd.step()
        losses_sgd.append(loss.item())

    model_adam = make_model()
    opt_adam = torch.optim.Adam(model_adam.parameters(),lr=0.01)
    losses_adam = []
    for _ in range(100):
        pred = model_adam(X)
        loss = nn.functional.mse_loss(pred,y)
        opt_adam.zero_grad();loss.backward();opt_adam.step()
        losses_adam.append(loss.item())

    print(f"✅ ex6 passed — SGD final loss: {losses_sgd[-1]:.4f}")
    print(f"              Adam final loss: {losses_adam[-1]:.4f}")

def ex7_optimizer_param_groups():
    """Use different LR for different parts of model."""
    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )

    opt = torch.optim.SGD([
        {'params': model[0].parameters(), 'lr': 0.001},
        {'params': model[2].parameters(), 'lr': 0.01},
    ], momentum=0.9)

    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]['lr'] == 0.001
    assert opt.param_groups[1]['lr'] == 0.01
    print(f"✅ ex7 passed — used different LR for different layers")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 12 Exercises: Optimizers")
    print("=" * 50 + "\n")

    ex1_sgd_update()
    ex2_momentum_exists()
    ex3_rmsprop()
    ex4_adam()
    ex5_adamw()
    ex6_convergence_speed()
    ex7_optimizer_param_groups()

    print("\n🎉 All Day 12 exercises complete!")
