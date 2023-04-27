import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# set up densenet model for cifar10
class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.densenet(x)


# load model from file
if os.path.exists('densenet_cifar10.pth'):
    model = DenseNet()
    model.load_state_dict(torch.load('densenet_cifar10.pth'))
    model.eval()
    print('Loaded model from file')
    test_dataset = datasets.CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# set up cifar10 dataset loaders
batch_size = 100

train_dataset = datasets.CIFAR10(root='data/', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = datasets.CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# train densenet on cifar10
def train(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print('Epoch: {}. Validation Accuracy: {} %'.format(epoch, 100 * correct / total))

    # save model
    torch.save(model.state_dict(), 'densenet_cifar10.pth')


model = DenseNet()
train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)
