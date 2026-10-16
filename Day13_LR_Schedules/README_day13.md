# Day 13: Learning Rate Schedules

## ✅ Completed
- [x] StepLR — drop every N epochs
- [x] MultiStepLR — drop at specific epochs
- [x] ExponentialLR — smooth decay
- [x] CosineAnnealingLR — modern default
- [x] ReduceLROnPlateau — reactive
- [x] OneCycleLR — warmup + cooldown
- [x] Compared 4 schedules on MNIST

## 📊 Schedules Table

| Schedule | Behavior | Best For |
|----------|----------|----------|
| StepLR | Drop every N epochs | Simple, predictable |
| MultiStepLR | Drop at milestones | Hand-tuned |
| ExponentialLR | Smooth decay | Continuous training |
| CosineAnnealing | Smooth curve | Modern default |
| ReduceLROnPlateau | Reactive | Safe choice |
| OneCycle | Rise + fall | Fast training |

## 🧩 PyTorch

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    train(...)
    scheduler.step()   # after optimizer.step()