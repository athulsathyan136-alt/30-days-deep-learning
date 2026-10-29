# Day 13: Learning Rate Schedules

## 🎯 Learning Objectives
- Why fixed LR is suboptimal
- Common schedules and when to use each
- PyTorch scheduler API
- Visualize LR over training

## 🧠 Why Schedules?

Fixed LR trade-off:
- High: fast but unstable
- Low: stable but slow

Solution: **start high, decay over time**

## 📊 Common Schedules

### 1. StepLR
lr = base_lr * gamma^(epoch // step_size)

text
Drop by factor `gamma` every `step_size` epochs.
Example: `gamma=0.1, step_size=30` → lr drops 10x every 30 epochs.

### 2. MultiStepLR
Drop at specific epochs. E.g. `milestones=[30, 60, 90]`.

### 3. ExponentialLR
lr = base_lr * gamma^epoch

text
Smooth decay every epoch.

### 4. CosineAnnealingLR
lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2

text
Smooth cosine curve. Popular in modern DL.

### 5. ReduceLROnPlateau
Drop LR when a metric (e.g. val loss) stops improving.
scheduler.step(val_loss)

text
Reactive — drops only when needed.

### 6. OneCycleLR
Warm up from low → high → low. Very popular for fast training.

## 🧩 PyTorch API

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    train(...)
    scheduler.step()   # update LR AFTER each epoch
📈 Visualizing LR
Always plot scheduler.get_last_lr() vs epoch to verify the schedule.

💡 Key Takeaways
LR schedules improve convergence

ReduceLROnPlateau is reactive and safe

CosineAnnealing is modern default

OneCycle for fast training

Always plot the LR curve to verify