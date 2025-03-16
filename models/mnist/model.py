import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# This is duplicated code pretty much, make a base class
class Model():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epoch = 0
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=config['learning_rate'] or 0.001, momentum=config.get('momentum', 0.9))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config['scheduler_patience'], factor=config['scheduler_factor'], verbose=True)
    
    def save(self, epoch, path):
        print(f"Saving model to {path}")
        torch.save({
            'config': self.config,
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
    
    def load(self, path):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.epoch = checkpoint['epoch']+1
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    def to(self, device):
        self.net.to(device)
