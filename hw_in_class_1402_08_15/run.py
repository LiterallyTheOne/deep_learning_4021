import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from my_model import MyNetwork


def train(trainloader, my_net, device):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(my_net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(my_net.parameters())

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = my_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(my_net.state_dict(), './cifar_net_2.pth')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train(trainloader, my_net, device)

    my_net = MyNetwork().to(device)
    my_net.load_state_dict(torch.load('./cifar_net_2.pth'))

    print(my_net)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = my_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Train accuracy: {100 * correct // total} %')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = my_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test accuracy: {100 * correct // total} %')


if __name__ == '__main__':
    main()
