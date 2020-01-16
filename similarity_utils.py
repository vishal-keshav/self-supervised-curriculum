import numpy as np
import torch


def calculate_distance(e1, e2):
    return np.linalg.norm(e1-e2)

def similar_embedding(id, nr_images=1000, k=6, model=None, test_dataset=None):
    # return the id list of 10 closes images
    res = []
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    # Create the embedding from the image id
    img, label = test_dataset[id]
    img = img[None, :, :, :]
    # print(img.size())
    # img = img.detach().numpy()[0] # Assume batch is of size 1
    embedding = model(img).detach().numpy()[0]
    distances = []
    for j, data in enumerate(test_loader):
        if j > nr_images:
            break
        img, label = data
        output = model(img)
        # print(output.size())
        output = output.detach().numpy()[0]
        dist = calculate_distance(embedding, output)
        distances.append(dist)
    res = np.array(distances).argsort()[:k]
    return res


def similar_pixels(id, nr_images=1000, k=6, test_dataset=None):
    # return the id list of 10 closes images
    res = []
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    # Create the embedding from the image id
    img, label = test_dataset[id]
    img_embedding = img.detach().numpy().ravel()
    distances = []
    for j, data in enumerate(test_loader):
        if j > nr_images:
            break
        img, label = data
        img = img.detach().numpy()[0].ravel()
        dist = calculate_distance(img_embedding, img)
        distances.append(dist)
    res = np.array(distances).argsort()[:k]
    return res
