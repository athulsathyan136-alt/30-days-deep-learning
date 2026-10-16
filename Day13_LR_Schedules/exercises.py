import torch
import torch.nn as nn


def ex1_step_lr():
    """StepLR drops LR every step_size epochs."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    lrs = []
    for epoch in range(10):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # LR should be: 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, ...
    assert abs(lrs[0] - 1.0) < 1e-6
    assert abs(lrs[3] - 0.5) < 1e-6
    assert abs(lrs[6] - 0.25) < 1e-6
    print(f"✅ ex1 passed — StepLR: {[round(l, 4) for l in lrs[:7]]}")


def ex2_multi_step_lr():
    """MultiStepLR drops at milestones."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6], gamma=0.1)

    lrs = []
    for epoch in range(10):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # Drops to 0.1 at epoch 3, 0.01 at epoch 6
    assert abs(lrs[0] - 1.0) < 1e-6
    assert abs(lrs[3] - 0.1) < 1e-6
    assert abs(lrs[6] - 0.01) < 1e-6
    print(f"✅ ex2 passed — MultiStepLR: {[round(l, 4) for l in lrs[:8]]}")


def ex3_exponential_lr():
    """ExponentialLR multiplies by gamma each epoch."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    lrs = []
    for epoch in range(10):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # 1.0, 0.9, 0.81, 0.729, ...
    assert abs(lrs[0] - 1.0) < 1e-6
    assert abs(lrs[1] - 0.9) < 1e-6
    assert abs(lrs[2] - 0.81) < 1e-6
    print(f"✅ ex3 passed — ExponentialLR: {[round(l, 4) for l in lrs[:5]]}")


def ex4_cosine_lr():
    """CosineAnnealingLR decreases like a cosine."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    lrs = []
    for epoch in range(11):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # Starts at 1.0, ends near 0
    assert lrs[0] > 0.99
    assert lrs[-1] < 0.02
    print(f"✅ ex4 passed — CosineAnnealingLR: first={lrs[0]:.4f}, last={lrs[-1]:.4f}")


def ex5_reduce_on_plateau():
    """ReduceLROnPlateau drops when metric plateaus."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Simulate val losses: keeps improving, then plateaus
    val_losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    for loss in val_losses:
        scheduler.step(loss)

    final_lr = scheduler.get_last_lr()[0]
    assert final_lr < 1.0
    print(f"✅ ex5 passed — ReduceLROnPlateau: lr dropped from 1.0 to {final_lr:.4f}")


def ex6_one_cycle():
    """OneCycleLR: rise then fall."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.0, total_steps=20
    )

    lrs = []
    for step in range(20):
        lrs.append(optimizer.param_groups[0]['lr'])
        # Note: OneCycleLR should be stepped AFTER each optimizer step, not epoch
        scheduler.step()

    # Should rise to max_lr, then fall
    max_lr_idx = lrs.index(max(lrs))
    assert max_lr_idx > 0
    assert lrs[0] < lrs[max_lr_idx]
    assert lrs[-1] < lrs[max_lr_idx]
    print(f"✅ ex6 passed — OneCycleLR peak at step {max_lr_idx}: {max(lrs):.4f}")


def ex7_scheduler_state():
    """Scheduler state can be saved and loaded."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for _ in range(5):
        scheduler.step()

    state = scheduler.state_dict()

    # New scheduler
    model2 = nn.Linear(10, 1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1.0)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.5)
    scheduler2.load_state_dict(state)

    # Both should be at the same point
    assert scheduler2.last_epoch == scheduler.last_epoch
    print(f"✅ ex7 passed — scheduler state saved & loaded (epoch {scheduler.last_epoch})")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 13 Exercises: Learning Rate Schedules")
    print("=" * 50 + "\n")

    ex1_step_lr()
    ex2_multi_step_lr()
    ex3_exponential_lr()
    ex4_cosine_lr()
    ex5_reduce_on_plateau()
    ex6_one_cycle()
    ex7_scheduler_state()

    print("\n🎉 All Day 13 exercises complete!")