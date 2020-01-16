#######
#   IMPORT
######

import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from feature_embedding_visualization.visualization import cluster

#######
#   DATASET
######

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                 download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

data_arr = []
label_arr = []

MAX_DATAPOINTS_TO_SHOW = 100

#######
#   LOOP
######

for i, data in enumerate(data_loader):

    if i >= MAX_DATAPOINTS_TO_SHOW:
        break

    label_arr.append(data[1])
    data_arr.append(data[0].flatten().tolist())


data_arr = np.array(data_arr)
label_arr = np.array(label_arr)

print(data_arr.shape)

print(data_arr.shape)
print(label_arr.shape)

cluster_obj = cluster(data_arr, label_arr)
cluster_obj.apply_tsne(n_components=2, perplexity=30)
cluster_obj.plot_cluster("toto.pdf")

# d = next(iter(data_loader))[0]
# print().size())
