import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

from scipy.special import softmax
import matplotlib.pyplot as plt
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# this variable will be used to avoid overwriting the weights of the other classifier.
# choose 1 for m2f classifier, 0 for Blond/Not Blond classifier.
parser = argparse.ArgumentParser()
parser.add_argument('--m2f', type=int, default=1, help="1 for m2f classifier 0 for BlondHair classifier ")
opts = parser.parse_args()
m2f_classifier = opts.m2f

batch_size = 4
learning_rate = 1e-3

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

if m2f_classifier:
    # this one is for the Eyeglasses remover domain transfer, it will train the Male/Female classifier.
    train_dataset = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/train', transform=transforms)
    test_dataset = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/test',transform=transforms)
    test_dataset_balanced = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/test_balanced', transform=transforms)
    test_dataset_unbalanced = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/test_unbalanced', transform=transforms)
else:
    # this one is for the m2f domain transfer, it will train the Blond/Not Blond classifier.
    train_dataset = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/Hair_classifier/train', transform=transforms)
    test_dataset = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/Hair_classifier/test', transform=transforms)
    test_dataset_balanced = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/Hair_classifier/test_balanced', transform=transforms)
    test_dataset_unbalanced = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/Hair_classifier/test_unbalanced', transform=transforms)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_dataloader_balanced = DataLoader(test_dataset_balanced, batch_size=batch_size)
test_dataloader_unbalanced = DataLoader(test_dataset_unbalanced, batch_size=batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def imshow(inp, title=None):
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


net = models.resnet50(pretrained=True)
net = net.cuda() if device else net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net.fc = net.fc.cuda() if device else net.fc

num_success = 0
n_epochs = 20
print_every = 10
last_good_epoch = 0
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
min_acc_diff = 0
for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()

        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target_).item()
        total += target_.size(0)
        if (batch_idx) % 4000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_dataloader))
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        network_learned = batch_loss < valid_loss_min

        if network_learned:
            valid_loss_min = batch_loss
            if m2f_classifier:
                torch.save(net.state_dict(), 'resnet_m2f.pt')
            else:
                torch.save(net.state_dict(), 'resnet_Blond.pt')
            print('Improvement-Detected, save-model')
            last_good_epoch = epoch
    net.train()


# we will calculate and save the probability here using the softmax function
prob_balanced = []
prob_unbalanced = []

if m2f_classifier:
    net.load_state_dict(torch.load('resnet_m2f.pt'))
else:
    net.load_state_dict(torch.load('resnet_Blond.pt'))

for data_t2, target_t2 in test_dataloader_balanced:
    data_t2, target_t2 = data_t2.to(device), target_t2.to(device)
    outputs_t2 = net(data_t2)
    prob_t2 = softmax(outputs_t2.cpu().detach().numpy(), axis=1)
    prob_balanced.append(prob_t2)

for data, target in test_dataloader_unbalanced:
    data, target = data.to(device), target.to(device)
    outputs = net(data)
    prob = softmax(outputs.cpu().detach().numpy(), axis=1)
    prob_unbalanced.append(prob)

if m2f_classifier:
    torch.save(prob_balanced, 'softmax_values_balanced_m2f.pt')
    torch.save(prob_unbalanced, 'softmax_values_unbalanced_m2f.pt')
else:
    torch.save(prob_balanced, 'softmax_values_balanced_blond.pt')
    torch.save(prob_unbalanced, 'softmax_values_unbalanced_blond.pt')
