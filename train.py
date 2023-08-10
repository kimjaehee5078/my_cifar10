import argparse
import torchvision
import os
import torch
from models.vgg import vgg11_bn
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


def train(opt):
    # Augumentation
    train_transforms = transforms.Compose([transforms.RandomCrop(32),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616))])
    
    val_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616))])
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # tensorboard settings
    log_dir = Path('logs')/name
    tb_writer = SummaryWriter(log_dir=log_dir)
 
    # Trian dataset
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck')
    num_workers = min([os.cpu_count(), batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    #Valldation dataset
    val_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Network model
    model = vgg11_bn()

    # GPU-support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:    # multi-GPU
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    #loading a weight file (if exists)
    #weight_file = Path('weights')/(name + '.pth')
    weight_file = Path('weights')/opt.weight
    best_accuracy = 0.0
    start_epoch, end_epoch = (0, epochs)
    if opt.resume and os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('resumed from epoch %d' % start_epoch)


    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch, end_epoch-1))
        t0 = time.time()
        epoch_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)/n' % (epoch_loss, t1-t0))
        lr_scheduler.step()
        
        # validation
        val_epoch_loss, accuracy = val_one_epoch(val_dataloader, model, loss_fn, device)
        print('[validation] loss=%.4f, accuracy=%.4f' %(val_epoch_loss, accuracy))
       

        #saving the best status into a weight file
        if accuracy > best_accuracy:
            best_weight_file = Path('weights')/(name + '_best.pth')
            best_accuracy = accuracy
            state = {'model' : model.state_dict(), 'epoch' : epoch, 'best_accuracy' : best_accuracy}
            torch.save(state, best_weight_file)
            print('best accuracy=>saved\n')

        #saving the current status into a weight file
        state = {'model' : model.state_dict(), 'epoch' : epoch, 'best_accuracy' : best_accuracy}
        torch.save(state, weight_file)
        # tensorboard logging
        tb_writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
        tb_writer.add_scalar('val_epoch_loss', val_epoch_loss, epoch)
        tb_writer.add_scalar('val_accuracy', accuracy, epoch)

def val_one_epoch(val_dataloader, model, loss_fn, device):
    model.eval()
    losess = []
    correct = 0
    total = 0
    for i, (imgs, targets) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            losess.append(loss.item())
            preds = torch.argmax(preds, axis=1)
            total += preds.size(0)
            correct += (preds == targets).sum().item()

    accuracy = correct/total
    return torch.tensor(losess).mean().item(), accuracy




def train_one_epoch(train_dataloader, model, loss_fn, optimizer, device):
    model.train()
    losses =[]
    for i, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad() # zeros the parameter gradients
        preds = model(imgs)  # forward
        loss = loss_fn(preds, targets) # calculates the iteration loss
        loss.backward()       # backwad
        optimizer.step()      # update weights
        # print the iteration loss every 100 iterations
        if i % 100 == 0:
            print('\t iteration: %d/%d, loee=%.4f' % (i, len(train_dataloader)-1, loss))
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=512, help='batch size')
    parser.add_argument('--name', '-n', default='ohhan', help='name for the run')
    parser.add_argument('--weight', '-w', default='ohhan.pth', help='weight file to be loaded')
    parser.add_argument('--resume', '-r', action='store_true', help='resume if this is true')


    opt = parser.parse_args()

    train(opt)

