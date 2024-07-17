import time
import wandb
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def process_data(num_workers):
    times = []
    train_dataloader = DataLoader(
        #,  # bring the dataset
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    for step, batch_data in enumerate(train_dataloader):
        start_time = time.time()
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        end_time = time.time()
        times.append(end_time - start_time)

    return times

num_workers_list = [4, 8, 12, 16, 20, 24, 28, 30]  # Example list of different numbers of workers
worker_times = {}

for num_workers in num_workers_list:
    print(f"Processing with {num_workers} workers...")
    times = process_data(num_workers)
    worker_times[num_workers] = times

average_times = {num_workers: sum(times) / len(times) for num_workers, times in worker_times.items()}

plt.figure()
plt.plot(list(average_times.keys()), list(average_times.values()), marker='o')
plt.xlabel('Number of Workers')
plt.ylabel('Average Time per Iteration (seconds)')
plt.title('Time vs Number of Workers')
plt.grid(True)
plt.show()

run = wandb.init(project="isic_lesions_24")
for num_worker, time_taken in zip(list(average_times.keys()), list(average_times.values())):
    wandb.log({"num_worker": num_worker, "time_taken": time_taken})
run.finish()
