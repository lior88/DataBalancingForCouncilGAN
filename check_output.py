import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time
import copy
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 4
learning_rate = 1e-3
num_pics = 12
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

parser = argparse.ArgumentParser()
parser.add_argument('--m2f', type=int, default=1, help="1 for m2f classifier 0 for BlondHair classifier ")
opts = parser.parse_args()
m2f_classifier = opts.m2f

if m2f_classifier:
    test_dataset_balanced = datasets.ImageFolder(
        root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/test_balanced', transform=transforms)
    test_dataset_unbalanced = datasets.ImageFolder(
        root=r'/home/danlior/Council-GAN-master/MaleFemale_classifier/test_unbalanced', transform=transforms)
else:
    test_dataset_balanced = datasets.ImageFolder(root=r'/home/danlior/Council-GAN-master/Hair_classifier/test_balanced',
                                                 transform=transforms)
    test_dataset_unbalanced = datasets.ImageFolder(
        root=r'/home/danlior/Council-GAN-master/Hair_classifier/test_unbalanced', transform=transforms)

test_dataloader_balanced = DataLoader(test_dataset_balanced, batch_size=batch_size)
test_dataloader_unbalanced = DataLoader(test_dataset_unbalanced, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if m2f_classifier:
    prob_unbalanced = torch.load('softmax_values_unbalanced_m2f.pt')
    prob_balanced = torch.load('softmax_values_balanced_m2f.pt')
    prob_balanced = np.reshape(prob_balanced, (-1, 2))
    prob_unbalanced = np.reshape(prob_unbalanced, (-1, 2))
    prob_balanced_male = prob_balanced[:, 1]
    prob_balanced_female = prob_balanced[:, 0]
    prob_unbalanced_male = prob_unbalanced[:, 1]
    prob_unbalanced_female = prob_unbalanced[:, 0]

    balanced_mean_male = np.mean(prob_balanced_male)
    unbalanced_mean_male = np.mean(prob_unbalanced_male)
    balanced_std_male = np.std(prob_balanced_male)
    unbalanced_std_male = np.std(prob_unbalanced_male)

    print("balanced male")
    print("mean: {}".format(balanced_mean_male))
    print("std: {}".format(balanced_std_male))
    print("unbalanced male")
    print("mean: {}".format(unbalanced_mean_male))
    print("std: {}".format(unbalanced_std_male))

    # special cases: try to find the num_pics images that the difference between the balanced and unbalanced probability is the highest
    # subtract the probability arrays and find the absolute highest values == largest distance
    diff = np.abs(prob_balanced_male - prob_unbalanced_male)
    diff_max = np.abs(prob_balanced_male - prob_unbalanced_male)
    idx = np.zeros(num_pics)
    for i in range(num_pics):
        idx[i] = np.argmax(diff_max)
        diff_max[int(idx[i])] = 0

    indexes = [int(i) for i in idx]

    print("special cases indexes: {}".format(indexes))
    print("special cases difference values. {}".format(diff[indexes]))
    print("male balanced special cases probabilities: {}".format(prob_balanced_male[indexes]))
    print("male not balanced special cases probabilities: {}".format(prob_unbalanced_male[indexes]))

    different_choice = sum(((prob_balanced_male > 0.5) & (prob_unbalanced_male < 0.5)) | (
                (prob_balanced_male < 0.5) & (prob_unbalanced_male > 0.5)))
    print("percent of pictures the decision was changed: {}".format(100*different_choice/len(prob_unbalanced_male)))
    male_different_choice = sum(((prob_balanced_male > 0.5) & (prob_unbalanced_male < 0.5)))
    print("percent of pictures that changed the decision for the better: {}".format(
        100 * float(male_different_choice / different_choice)))
else:
    prob_unbalanced = torch.load('softmax_values_unbalanced_blond.pt')
    prob_balanced = torch.load('softmax_values_balanced_blond.pt')
    prob_balanced = np.reshape(prob_balanced[:-1], (-1, 2))
    prob_unbalanced = np.reshape(prob_unbalanced[:-1], (-1, 2))
    prob_balanced_not_blond = prob_balanced[:, 1]
    prob_balanced_blond = prob_balanced[:, 0]
    prob_unbalanced_not_blond = prob_unbalanced[:, 1]
    prob_unbalanced_blond = prob_unbalanced[:, 0]

    balanced_mean_not_blond = np.mean(prob_balanced_not_blond)
    unbalanced_mean_not_blond = np.mean(prob_unbalanced_not_blond)
    balanced_std_not_blond = np.std(prob_balanced_not_blond)
    unbalanced_std_not_blond = np.std(prob_unbalanced_not_blond)

    print("balanced not blond")
    print("mean: {}".format(balanced_mean_not_blond))
    print("std: {}".format(balanced_std_not_blond))
    print("unbalanced not blond")
    print("mean: {}".format(unbalanced_mean_not_blond))
    print("std: {}".format(unbalanced_std_not_blond))

    # special cases: try to find the num_pics images that the difference between the balanced and unbalanced probability is the highest
    # subtract the probability arrays and find the absolute highest values == largest distance
    diff = np.abs(prob_balanced_not_blond - prob_unbalanced_not_blond)
    diff_max = np.abs(prob_balanced_not_blond - prob_unbalanced_not_blond)
    idx = np.zeros(num_pics)
    for i in range(num_pics):
        idx[i] = np.argmax(diff_max)
        diff_max[int(idx[i])] = 0

    indexes = [int(i) for i in idx]

    print("special cases indexes: {}".format(indexes))
    print("special cases difference values. {}".format(diff[indexes]))
    print("not Blond balanced special cases probabilities: {}".format(prob_balanced_not_blond[indexes]))
    print("not Blond not balanced special cases probabilities: {}".format(prob_unbalanced_not_blond[indexes]))


    different_choice = sum(((prob_balanced_not_blond > 0.5) & (prob_unbalanced_not_blond < 0.5)) | (
                (prob_balanced_not_blond < 0.5) & (prob_unbalanced_not_blond > 0.5)))
    print("percent of pictures the decision was changed: {}".format(100 * different_choice/len(prob_unbalanced_blond)))
    not_blond_different_choice = sum(((prob_balanced_not_blond > 0.5) & (prob_unbalanced_not_blond < 0.5)))
    print("percent of pictures that changed the decision for the better: {}".format(
        100 * float(not_blond_different_choice / different_choice)))

plt.figure()
i = 1
for a in indexes:
    ax = plt.subplot(3, 4, i)
    pic = test_dataset_balanced.imgs[a]
    img = mpimg.imread(pic[0])
    ax.imshow(img)
    plt.axis('off')
    i = i + 1
plt.suptitle('balanced')


plt.figure()
i = 1
for a in indexes:
    ax = plt.subplot(3, 4, i)
    pic = test_dataset_unbalanced.imgs[a]
    img = mpimg.imread(pic[0])
    ax.imshow(img)
    plt.axis('off')
    i = i + 1
plt.suptitle('unbalanced')
plt.show()