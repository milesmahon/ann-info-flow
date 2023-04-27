import torch
from torchvision import datasets
from torchvision.transforms import transforms

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)

# change the last layer to 100 classes for cifar100
model.classifier[6] = torch.nn.Linear(4096, 100)

# set up cifar100 dataset loaders with resizing for alexnet
batch_size = 100
train_dataset = datasets.CIFAR100(root='data/', train=True, transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
]), download=True)
val_dataset = datasets.CIFAR100(root='data/', train=False, transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
]), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# train alexnet on cifar100
def train(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        top_5_correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # get top-5 accuracy
            _, predicted = torch.topk(output.data, 5, 1)
            # check if target is in top-5 predictions
            top_5_correct += (predicted == target.view(-1, 1).expand_as(predicted)).sum().item()

        print('Epoch: %d, Accuracy: %d %%' % (epoch + 1, 100 * correct / total))
        print('Epoch: %d, Top-5 Accuracy: %d %%' % (epoch + 1, 100 * top_5_correct / total))

    # save model
    torch.save(model.state_dict(), 'alexnet_cifar100_raw.pth')


train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)


# test alexnet on cifar100
def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# test_model(model, val_loader)
