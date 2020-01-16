import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms


from model import Model
import similarity_utils

dataset = torchvision.datasets.CIFAR10

transform = transforms.Compose([
    transforms.ToTensor()
])

valid_dataset = dataset(root='./data', train=False, download=True, transform=transform)




model = Model()
model.load_state_dict(torch.load('model.pt'))
model.eval()

a = similarity_utils.similar_embedding(600, 1000, 6, model, valid_dataset)

fig, ax = plt.subplots(1, 6)

fig = plt.figure()

plt.subplot(2, 6, 1)
plt.imshow(valid_dataset[a[0]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 2)
plt.imshow(valid_dataset[a[1]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 3)
plt.imshow(valid_dataset[a[2]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 4)
plt.imshow(valid_dataset[a[3]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 5)
plt.imshow(valid_dataset[a[4]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 6)
plt.imshow(valid_dataset[a[5]][0].numpy().transpose((1, 2, 0)))

a = similarity_utils.similar_pixels(600, 1000, 6, valid_dataset)

plt.subplot(2, 6, 7)
plt.imshow(valid_dataset[a[0]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 8)
plt.imshow(valid_dataset[a[1]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 9)
plt.imshow(valid_dataset[a[2]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 10)
plt.imshow(valid_dataset[a[3]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 11)
plt.imshow(valid_dataset[a[4]][0].numpy().transpose((1, 2, 0)))
plt.subplot(2, 6, 12)
plt.imshow(valid_dataset[a[5]][0].numpy().transpose((1, 2, 0)))

plt.savefig('fig3.pdf')

