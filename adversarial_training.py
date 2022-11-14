'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# data path
parser.add_argument('--data_path', default='./data', type=str, help='data folder path')

# training parameter
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_false',
                    help='resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1024, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    best_acc = 0
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

upper = torch.tensor([1-0.4914, 1-0.4822, 1-0.4465])
upper /= torch.tensor([0.2023, 0.1994, 0.2010])
lower = torch.tensor([-0.4914, -0.4822, -0.4465])
lower /= torch.tensor([0.2023, 0.1994, 0.2010])


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain in the range
    # transpose the batch and channel dimension
    perturbed_image = perturbed_image.transpose(0,1)
    # perturbed_image = perturbed_image.detach().cpu().numpy()
    for channel, _ in enumerate(perturbed_image):
        u_limit, l_limit = upper[channel], lower[channel]
        # perturbed_image[channel] = torch.clamp(perturbed_image[channel].data, l_limit, u_limit)  
        perturbed_image[channel] = perturbed_image[channel].data.clamp(l_limit, u_limit)
    perturbed_image = perturbed_image.transpose(0,1)
    # Return the perturbed image
    return perturbed_image


def test_fgsm(model, device, data, target, epsilon=0.05):

    
    # Send the data and label to the device
    #data, target = data.to(device), target.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True
    # Forward pass the data through the model
    output = model(data)

    # Calculate the loss
    loss = criterion(output, target)
    # Zero all existing gradients
    model.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data
    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    #perturbed_data = data
    # Return the accuracy and an adversarial example

    return perturbed_data




# train the model
def train_one_epoch(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total, correct_adv, total_adv = 0,0,0,0,0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # normal training
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # adversarial training
        # adv_input = inputs
        adv_input = test_fgsm(net, device, inputs, targets)#.to(device)
        adv_input = adv_input.clone().detach().to(device)
        optimizer.zero_grad()
        outputs = net(adv_input)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_adv += targets.size(0)
        correct_adv += predicted.eq(targets).sum().item()



        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct_adv/total_adv, correct_adv, total_adv))

# test the accuracy of the model
def test(epoch):
    global best_acc
    #net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_adv, correct_adv = 0,0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        # adversarial training
        # adv_input = inputs
        adv_input = test_fgsm(net, device, inputs, targets)#.to(device)
        adv_input = adv_input.clone().detach().to(device)
        optimizer.zero_grad()
        outputs = net(adv_input)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total_adv += targets.size(0)
        correct_adv += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct_adv/total_adv, correct_adv, total_adv))

    # Save checkpoint.
    acc = 100.*correct/total
    acc += 100.*correct_adv/total_adv
    acc = acc/2
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_adv_scratch.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train_one_epoch(epoch)
    test(epoch)
    scheduler.step()
