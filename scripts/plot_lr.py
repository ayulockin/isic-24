import torch
import matplotlib.pyplot as plt

T_0 = 100
T_mult = 2
eta_max = 0.01
eta_min = 1e-6

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=626,
    T_mult=4,
    eta_min=1e-6,
    last_epoch=-1
)

lrs = []
steps = 0
for epoch in range(50):
    for step in range(626):
        steps += 1
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

# Plotting the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(range(steps), lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Warm Restarts (New Parameters)')
plt.grid(True)
plt.savefig('plt.png')
