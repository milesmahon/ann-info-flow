import os.path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# check for gpu
from analyze_info_flow import analyze_info_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.isfile("alexnet_cifar100_pre_froze.pth"):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 100)
    model.load_state_dict(torch.load("alexnet_cifar100_pre_froze.pth", map_location=device))
    model.eval()
    print("Loaded model from file")
    test_dataset = datasets.CIFAR100(root='data/', train=False, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))
