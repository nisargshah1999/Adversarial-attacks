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

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# data path
parser.add_argument('--data_path', default='./data', type=str, help='data folder path')

# training parameter
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_false',
                    help='resume from checkpoint')

# experiment path
parser.add_argument('--output_path', default='./result', type=str, help='result folder path')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

upper = torch.tensor([1-0.4914, 1-0.4822, 1-0.4465])
upper /= torch.tensor([0.2023, 0.1994, 0.2010])
lower = torch.tensor([-0.4914, -0.4822, -0.4465])
lower /= torch.tensor([0.2023, 0.1994, 0.2010])

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4096, shuffle=False, num_workers=2)

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
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# results path
try:
    os.mkdir(args.output_path)    
except  FileExistsError:
    pass
try:
    os.mkdir(os.path.join(args.output_path,'results_fgsm'))    
except  FileExistsError:
    pass

net.eval()

# CW attack code
def cw_attack(model, images, labels, device, c=1, kappa=0, steps=50, lr=0.01):

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    cov = torch.tensor([0.2023, 0.1994, 0.2010])
    images = images.transpose(0,1)
    for i in range(3):
        images[i] = images[i]*cov[i]+mean[i]
    images = images.transpose(0,1)
    
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    w = inverse_tanh_space(images).detach()
    w.requires_grad = True

    MSELoss = nn.MSELoss(reduction='none')
    Flatten = nn.Flatten()

    optimizer = optim.Adam([w], lr=lr)

    for step in range(steps):
        # Get adversarial images
        adv_images = tanh_space(w)
        adv_images = adv_images.transpose(0,1)
        for i in range(3):
            adv_images[i] = (adv_images[i]-mean[i])/cov[i]
        adv_images = adv_images.transpose(0,1)

        # Calculate loss
        current_L2 = MSELoss(Flatten(adv_images),
                                Flatten(images)).sum(dim=1)
        L2_loss = current_L2.sum()

        outputs = model(adv_images)
        f_loss = f(outputs, labels, device, kappa).sum()
        loss = L2_loss + c*f_loss

        optimizer.zero_grad()
        model.zero_grad
        loss.backward()
        optimizer.step()

    return adv_images


def tanh_space(x):
    return 1/2*(torch.tanh(x) + 1)

def inverse_tanh_space(x):
    return atanh(x*2-1)

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

# f-function in the paper
def f(outputs, labels, device, kappa):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

    return torch.clamp((j-i), min=-kappa)



def test_cw(model, device, test_loader, c=1, kappa=0):
    # Accuracy counter
    correct, total = 0,0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model
        output = model(data)
        _ , init_pred = output.max(1) # get the index of the max log-probability

        # Call CW Attack
        perturbed_data = cw_attack(model, data, target, device, c=c, kappa=kappa, steps=50, lr=0.01)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        _, final_pred = output.max(1)
        total += target.size(0)
        correct += final_pred.eq(target).sum().item()

        # Save some adv examples for visualization
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((init_pred, final_pred, adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("c: {} and kappa: {}\tTest Accuracy = {} / {} = {}".format(c, kappa, correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# Iterating across different epsilons values
accuracies = []
examples = []

# Run test for each epsilon
# for eps in epsilons[:]:
box_constraint = [1e-2, 1e-1, 1, 1e1, 1e2]
for c in box_constraint:
    acc, ex = test_cw(net, device, testloader, c=c, kappa=0)
    accuracies.append(acc)
    examples.append(ex)

counter = 0
mean, cov = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
for index, c in enumerate(box_constraint):
    for batch in examples[index]:     
        for i in range(len(batch[0])):
            init_pred = batch[0][i]
            final_pred = batch[1][i]
            adv_ex = batch[2][i]
            adv_ex = np.clip(adv_ex.transpose(1,2,0)*cov+mean,0,1)
            plt.imsave(f'{args.output_path}/results_cw/{c}_{init_pred}_{final_pred}.png',adv_ex)