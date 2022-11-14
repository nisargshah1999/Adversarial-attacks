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
epsilons = [0, .05, .1, .15, .2, .25, .3]

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain in the range
    # transpose the batch and channel dimension
    perturbed_image = perturbed_image.transpose(0,1)
    for channel, _ in enumerate(perturbed_image):
        u_limit, l_limit = upper[channel], lower[channel]
        perturbed_image[channel] = torch.clamp(perturbed_image[channel], l_limit, u_limit)        
    perturbed_image = perturbed_image.transpose(0,1)
    # Return the perturbed image
    return perturbed_image

def test_fgsm(model, device, test_loader, epsilon):

    # Accuracy counter
    correct, total = 0,0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        _ , init_pred = output.max(1) # get the index of the max log-probability

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
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # performance under the attack
        _, final_pred = output.max(1)
        total += target.size(0)
        correct += final_pred.eq(target).sum().item()

        # Save some adv examples for visualization
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((init_pred, final_pred, adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# Iterating across different epsilons values
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons[:]:
    acc, ex = test_fgsm(net, device, testloader, eps)
    accuracies.append(acc)
    examples.append(ex)

counter = 0
mean, cov = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
for index, eps in enumerate(epsilons):
    for batch in examples[index]:      
        for i in range(len(batch[0])):
            init_pred = batch[0][i]
            final_pred = batch[1][i]
            adv_ex = batch[2][i]
            adv_ex = np.clip(adv_ex.transpose(1,2,0)*cov+mean,0,1)
            plt.imsave(f'{args.output_path}/results_fgsm/{eps}_{init_pred}_{final_pred}.png',adv_ex)