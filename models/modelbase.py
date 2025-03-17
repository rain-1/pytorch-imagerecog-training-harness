import torch
import torch.nn as nn
import torch.optim as optim

class ModelBase():
    def __init__(self, net, config):
        super().__init__()
        self.config = config
        self.epoch = 0
        self.net = net
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
