import torch
import torchvision


def load_xor_data():
    x = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])

    return [x, y]


def load_MNIST_data():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10, shuffle=True,)

    return train_loader, test_loader
