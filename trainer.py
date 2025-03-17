import torch
from tqdm import tqdm
import wandb
from itertools import islice

DEBUG_TESTLOADER_SUBSET = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model, trainloader, testloader, use_wandb=False):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.use_wandb = use_wandb

    def train(self, epoch):
        running_counter = 0
        running_loss = 0.0
        for i, data in tqdm(enumerate(self.trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model.net(inputs)
            loss = self.model.criterion(outputs, labels)
            loss.backward()
            self.model.optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_counter += 1
            wandb.log({'loss': loss})

        if self.use_wandb:
            wandb.log({'train_loss': running_loss / running_counter})

    def test(self, epoch):
        val_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            # Only evaluate on a fraction of validation data

            if DEBUG_TESTLOADER_SUBSET:
                subset_size = 0.2  # Evaluate on 20% of validation data
                testloader_subset = list(islice(self.testloader, int(len(self.testloader) * subset_size)))
            else:
                testloader_subset = self.testloader

            for inputs, labels in tqdm(testloader_subset):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model.net(inputs)
                loss = self.model.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(self.testloader)

        print(f"Test accuracy: {val_accuracy:.2f}%")

        if self.use_wandb:
            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch})
        
        return val_accuracy
