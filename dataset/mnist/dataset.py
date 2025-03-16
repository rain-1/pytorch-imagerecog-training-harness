import torch
import torchvision
import torchvision.transforms as transforms

classes = tuple(str(i) for i in range(10))  # MNIST classes are digits 0-9

batch_size = 32

train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])  # MNIST is grayscale, so only one channel
test_transform = transforms.Compose(
    [transforms.Resize((28, 28)),  # Rescale to MNIST image size
     transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
