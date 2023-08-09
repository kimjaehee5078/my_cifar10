import argparse
import torchvision
import os
import torch
from models.vgg import vgg11_bn


def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name

    
    # Trian dataset
    transforms = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transforms)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck')
    num_workers = min([os.cpu_count(), batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Network model
    model = vgg11_bn()
    pass # debug checkpoint

    for i, (imgs, targets) in enumerate(train_dataloader):
        print('batch idx=%d/%d' %(i, len(train_dataloader)-1))
        pass # debug checkpoint

    pass # debug breakpoint


    print(epochs)
    print(batch_size)
    print(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=200, help='batch size')
    parser.add_argument('--name', default='ohhan', help='name for the run')

    opt = parser.parse_args()

    train(opt)

