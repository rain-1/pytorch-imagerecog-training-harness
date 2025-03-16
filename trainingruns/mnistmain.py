import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from dataset.mnist.dataset import trainloader, testloader
from models.mnist.model import Model
from utilities.load_latest_checkpoint import load_latest_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network_run_name = "mnist-example"

# Initialize wandb
wandb.init(project=f"{network_run_name}", name="run-1")

# Define hyperparameters
config = {
    "epochs": 10,
    "learning_rate": 0.001,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5
}

wandb.config.update(config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from nntest import nntest
from nntrain import nntrain

model = Model(config)
model.to(device)

# Call the function
load_latest_checkpoint(network_run_name, model)

# Log model architecture
wandb.watch(model.net, log="all")

start_epoch = model.epoch
epoch = 0
for i in range(100):  # loop over the dataset multiple times
    epoch = start_epoch + i

    nntrain(epoch, model, use_wandb=True)
    model.save(epoch, f'checkpoints/{network_run_name}_net_{epoch}.pth')

    val_accuracy = nntest(epoch, model, use_wandb=True)
    model.scheduler.step(val_accuracy)
    current_lr = model.scheduler.get_last_lr()[0]
    wandb.log({'learning_rate': current_lr, 'epoch': epoch})
