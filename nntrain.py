import torch

from tqdm import tqdm
import wandb

# from dataset.cifar10.dataset import trainloader, testloader
# from models.cifar10tutorial.model import Model

from dataset.mnist.dataset import trainloader, testloader
#from models.cifar10tutorial.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nntrain(epoch, model, use_wandb=False):
    running_counter = 0
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        model.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.net(inputs)
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_counter += 1
        wandb.log({'loss': loss})

    if use_wandb:
        wandb.log({'train_loss': running_loss / running_counter})

# if __name__ == "__main__":
#     model = Model()
#     model.to(device)

#     PATH = f'checkpoints/cifar_net_0.pth'
#     model.save(0, PATH)

#     epoch = 0
#     for epoch in range(2):  # loop over the dataset multiple times
#         nntrain(epoch, model)

#     PATH = f'checkpoints/cifar_net_{epoch}.pth'
#     model.save(epoch, PATH)
