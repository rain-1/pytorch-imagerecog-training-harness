import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from utilities.load_latest_checkpoint import load_latest_checkpoint

from trainer import Trainer

########
from models.cifar10real.model import Model
import dataset.cifar10.dataset as dataset
network_run_name = "cifar10-real"

# Define hyperparameters
config = {
    "epochs": 10,
    "learning_rate": 0.001,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5
}
########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize wandb
wandb.init(project=f"{network_run_name}", name="run-1")

wandb.config.update(config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Model(config)
model.to(device)


trainer = Trainer(model, dataset.trainloader, dataset.testloader, use_wandb=True)

# Call the function
load_latest_checkpoint(network_run_name, model)

# Log model architecture
wandb.watch(model.net, log="all")

start_epoch = model.epoch
epoch = 0
for i in range(100):  # loop over the dataset multiple times
    epoch = start_epoch + i

    trainer.train(epoch)
    model.save(epoch, f'checkpoints/{network_run_name}_{epoch}.pth')

    val_accuracy = trainer.test(epoch)
    model.scheduler.step(val_accuracy)
    current_lr = model.scheduler.get_last_lr()[0]
    wandb.log({'learning_rate': current_lr, 'epoch': epoch})
