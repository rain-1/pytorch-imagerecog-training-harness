import torch

from dataset.cifar10.dataset import trainloader, testloader
from models.cifar10tutorial.model import Model

from itertools import islice

from tqdm import tqdm
import wandb

DEBUG_TESTLOADER_SUBSET = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nntest(epoch, model, use_wandb=False):
    val_loss = 0
    total = 0
    correct = 0
    
    with torch.no_grad():
        # Only evaluate on a fraction of validation data

        if DEBUG_TESTLOADER_SUBSET:
            subset_size = 0.2  # Evaluate on 20% of validation data
            testloader_subset = list(islice(testloader, int(len(testloader) * subset_size)))
        else:
            testloader_subset = testloader

        for inputs, labels in tqdm(testloader_subset):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.net(inputs)
            loss = model.criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(testloader)

    print(f"Test accuracy: {val_accuracy:.2f}%")

    if use_wandb:
        wandb.log({'val_loss': avg_val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch})
    
    return val_accuracy

if __name__ == "__main__":
    model = Model()
    model.load("checkpoints/cifar_net.pth")
    model.to(device)

    nntest(model)
