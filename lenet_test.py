import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# implement lenet model
from analyze_info_flow import analyze_info_flow


class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# train lenet on cifar100
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

            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, batch_idx + 1, len(train_loader), loss.item()))

        # validate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
            if correct / total > .85:
                break


    # save model
    torch.save(model.state_dict(), 'lenet_cifar100.pth')


if os.path.exists('lenet_cifar100.pth'):
    model = LeNet()
    model.load_state_dict(torch.load('lenet_cifar100.pth'))
    model.eval()
    analyze_info_flow(model,
                      {"dataset":datasets.CIFAR100(root='data/', train=False, transform=transforms.ToTensor(), download=True)},
                      {"num_train": 1000, "info_method":"corr"})
else:
    # setup cifar10 dataset loaders
    batch_size = 100
    train_dataset = datasets.CIFAR100(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    val_dataset = datasets.CIFAR100(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # train lenet
    model = LeNet()
    train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
