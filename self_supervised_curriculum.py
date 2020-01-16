device = 'cuda'

import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

##########################################################
# Tensor Utilities
def tensor_to_img(tensor):
    return tensor.cpu()[0].permute(1, 2, 0).numpy()

def plot_all(img1, img2, img3, img4):
    fig = plt.figure()
    ax = plt.subplot(1,4, 1)
    plt.imshow(img1)
    ax = plt.subplot(1,4, 2)
    plt.imshow(img2)
    ax = plt.subplot(1,4, 3)
    plt.imshow(img3)
    ax = plt.subplot(1,4, 4)
    plt.imshow(img4)

##########################################################
# Utility functions
def next_permu(e):
    l = list(e)
    i = len(l)-2
    while i>=0 and l[i]>l[i+1]:
        i-=1
    if i == -1:
        return tuple(l[::-1])
    j = len(l)-1
    while j>i and l[i]>l[j]:
        j-=1
    l[i],l[j] = l[j],l[i]
    l[i+1:] = l[-1:i:-1]
    return tuple(l)

def get_permutations(n):
    result = {}
    rank = 0
    start_permutation = tuple([i for i in range(n)])
    result[start_permutation] = rank
    rank+=1
    next_permutation = next_permu(start_permutation)
    while next_permutation != start_permutation:
        result[next_permutation] = rank
        rank+=1
        next_permutation = next_permu(next_permutation)
    return result

def get_refined_permutations(n):
    permutation_rank_refined = {}
    permutation_rank = get_permutations(n)
    for permu, rank in permutation_rank.items():
        if rank%2 == 0:
            permutation_rank_refined[rank/2] = permu
    return permutation_rank_refined

###########################################################

class CutImage(object):
    def __init__(self, divide = 4):
        self.divide = divide

    def __call__(self, sample):
        assert sample.shape == (3, 96, 96), print(sample.shape)
        img_list = [sample[:, 0:48, 0:48],
                    sample[:, 0:48, 48:96],
                    sample[:, 48:96, 0:48],
                    sample[:, 48:96, 48:96]]
        return img_list

class Rescale(object):
    def __init__(self, output_size = (96,96)):
        self.output_size = output_size
        self.transform_PIL = torchvision.transforms.ToPILImage()
        self.transform_resize = torchvision.transforms.Resize(size = self.output_size)
        self.transform_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        img = self.transform_PIL(img)
        img = self.transform_resize(img)
        img = self.transform_tensor(img)
        return img

###############################################################################

# Dataset for jigsaw solver
class JigsawPuzzleDataset(Dataset):
    def __init__(self, dataset, transform = None, train = "train",
                 image_transform = [], patch_transform = [], rescale = True):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        self.dataset = dataset(root='./data', split = train,
                               download = True, transform = transform)
        self.permu = get_refined_permutations(4)
        self.cut_img = CutImage(4)
        self.image_transform = image_transform
        self.patch_transform = patch_transform
        self.rescale = rescale
        self.rescale_transform = Rescale()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx][0]
        label = random.randint(0,11)
        p = self.permu[label]
        for t in self.image_transform:
            img = t(img)
        img_list = self.cut_img(img)
        for t in self.patch_transform:
            img_list = t(img_list)
        #img1 = img_list[0].transpose(2,0,1)
        if self.rescale:
            for i in range(4):
                img_list[i] = self.rescale_transform(img_list[i])
        img1 = img_list[p[0]].float().to(device)
        img2 = img_list[p[1]].float().to(device)
        img3 = img_list[p[2]].float().to(device)
        img4 = img_list[p[3]].float().to(device)
        return img1, img2, img3, img4, label

################################################################################

class PatchJitter(object):
    def __init__(self, jitter_scale = 0.95):
        self.jitter_scale = jitter_scale

    def randomCrop(self, img_tensor):
        C,H,W = img_tensor.size()
        H_new, W_new = int(H*self.jitter_scale), int(W*self.jitter_scale)
        x = np.random.randint(0, H-H_new)
        y = np.random.randint(0, W-W_new)
        return img_tensor[:, x:x+H_new, y:y+W_new]

    def __call__(self, img_list):
        jittered_list = []
        for img in img_list:
            jittered_list.append(self.randomCrop(img))
        return jittered_list

################################################################################
def print_sample_images_with_high_jitter():
    patch_jitter = PatchJitter(0.65)
    dataset = torchvision.datasets.STL10
    train_dataset = JigsawPuzzleDataset(dataset,
                        image_transform = [],
                        patch_transform = [patch_jitter])
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                  shuffle=False, num_workers=0)

    for i,data in enumerate(data_loader):
        img1, img2, img3, img4, label = data
        print(img1.shape)
        plot_all(tensor_to_img(img1), tensor_to_img(img2),
                          tensor_to_img(img3), tensor_to_img(img4))
        break

def print_sample_images_with_low_jitter():
    # print the sample images from jigsaw dataset after applying varying amount of jitter
    patch_jitter = PatchJitter(0.95)
    dataset = torchvision.datasets.STL10
    train_dataset = JigsawPuzzleDataset(dataset,
                        image_transform = [],
                        patch_transform = [patch_jitter])
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                  shuffle=False, num_workers=0)

    for i,data in enumerate(data_loader):
        img1, img2, img3, img4, label = data
        print(img1.shape)
        plot_all(tensor_to_img(img1), tensor_to_img(img2),
                          tensor_to_img(img3), tensor_to_img(img4))
        break
################################################################################

# Model architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )

        self.fc_proxy = nn.Sequential(
            nn.Linear(294912, 1024),
            nn.ReLU(inplace=False),

            nn.Linear(1024, 12)
        )

        self.fc_classifier = nn.Sequential(
            nn.Linear(73728, 1024),
            nn.ReLU(inplace=False),

            nn.Linear(1024, 10)
        )

    def forward(self, x1, x2=None, x3=None, x4=None, proxy=False):

        if proxy == False:
            output1 = self.cnn(x1)
            output = output1.view(output1.size(0), -1)
            output = self.fc_classifier(output)
            return output
        else:
            if x2 is not None:
                output1 = self.cnn(x1)
                output2 = self.cnn(x2)
                output3 = self.cnn(x3)
                output4 = self.cnn(x4)

                output1 = output1.view(output1.size(0), -1)
                output2 = output2.view(output2.size(0), -1)
                output3 = output2.view(output3.size(0), -1)
                output4 = output2.view(output4.size(0), -1)

                output = torch.cat((output1, output2, output3, output4), dim=1)

                output = self.fc_proxy(output)
                return output
            else:
                output1 = self.cnn(x1)
                output = output1.view(output1.size(0), -1)
                return output

###############################################################################
def train_validate_pretext_task(model_name = 'model.pt', nr_epochs = 1,
                                  image_transform = [], patch_transform = []):
    model = Model().to(device)
    if os.path.isfile(model_name):
        print ("Model file exists, retriving the weights")
        model.load_state_dict(torch.load(model_name))
    else:
        print ("Model file does not exists")

    dataset = torchvision.datasets.STL10

    train_dataset = JigsawPuzzleDataset(dataset, train = "unlabeled",
                              image_transform = image_transform,
                              patch_transform = patch_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                              shuffle=True, num_workers=0)

    valid_dataset = JigsawPuzzleDataset(dataset, train="test",
                              image_transform = image_transform,
                              patch_transform = patch_transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64,
                                              shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    losses = []
    val_acc = []

    for epoch in range(0, nr_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, nr_epochs))
        correct = 0
        total = 0
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            img1, img2, img3, img4, label = data
            img1, img2, img3, img4, label = img1.to(device), img2.to(device), img3.to(device), img4.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img1, img2, img3, img4, proxy = True)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss_contrastive = criterion(output, label)
            loss_contrastive.backward()
            optimizer.step()
            epoch_loss += loss_contrastive
        losses.append(epoch_loss / i)
        print('Training Loss: {}'.format(epoch_loss / i))
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            for j, data_valid in enumerate(valid_loader):
                img1, img2, img3, img4, label = data_valid
                img1, img2, img3, img4, label = img1.to(device), img2.to(device), img3.to(device), img4.to(device), label.to(device)
                answer = model(img1, img2, img3, img4, proxy=True)
                correct_predictions += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                total_predictions += (label == label).sum().item()
        val_acc.append((100 * correct_predictions / total_predictions))
        print('Accuracy of the network on train images: %d %%' % (
        100 * correct / total))
        print('Accuracy of the network on test images: %d %%' % (
        100 * correct_predictions / total_predictions))
        print('{} / {}'.format(correct_predictions, total_predictions))

    torch.save(model.state_dict(),model_name)
    return losses, val_acc

################################################################################
def curriculum_learning():
    jitter_levels = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    for jitter in jitter_levels:
        patch_transform = []
        if jitter != 1.0:
            patch_transform = [PatchJitter(jitter)]
        print("Curriculum step with jitter " + str(jitter))
        train_validate_pretext_task(model_name = 'model_curriculum.pt', nr_epochs = 1,
                                  image_transform = [], patch_transform = patch_transform)

###############################################################################
def train_validate_downstream_task(model_name = 'model.pt', nr_epochs = 10):
    model = Model().to(device)
    model.load_state_dict(torch.load(model_name))

    dataset = torchvision.datasets.STL10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])
    train_dataset = dataset(root='./data', split = "train", download=True,
                                                          transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=True, num_workers=0)

    valid_dataset = dataset(root='./data', split="test", download=True, transform=transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64,
                                           shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def get_acc_val(model):
        with torch.no_grad():
            true_positif = 0
            total_positif = 0
            for j, data_valid in enumerate(valid_loader):
                img, label = data_valid
                img, label = img.cuda(), label.cuda()
                answer = model(img)
                true_positif += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                total_positif += (label == label).sum().item()
        return true_positif, total_positif

    loss_total = []
    val_acc_saved_model = []

    for n_epoch in range(0, nr_epochs):

        print('Epoch: {}/{}'.format(n_epoch + 1, nr_epochs))
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        print('Training Loss: {}'.format(epoch_loss / i))
        loss_total.append(epoch_loss / i)
        true_positif, total_positif = get_acc_val(model)
        val_acc =  true_positif / total_positif
        val_acc_saved_model.append(val_acc)
        print('Accuracy of the network on test images: %d %%' % (
        100 * true_positif / total_positif))
        print('Test accuracy {} / {}'.format(true_positif, total_positif))
    return loss_total, val_acc_saved_model

################################################################################

def main():
    print("Training pretext task with curriculum learning")
    curriculum_learning()
    print("Training downstream task")
    train_validate_downstream_task(model_name = 'model_curriculum.pt', nr_epochs = 10)

main()
