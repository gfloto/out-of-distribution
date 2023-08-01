import sys, os 
import torch 
import numpy as np
import matplotlib.pyplot as plt

from percept import Percept
from datasets import get_loader

# given dict of loaders, sample one batch from each
def sample_one(loaders):
    samples = {}
    for dataset, loader in loaders.items():
        for x, _ in loader:
            x = 255 * x[0].permute(1,2,0).cpu().numpy()
            x = x.astype(np.uint8)
            samples[dataset] = x
            break

    return samples

# compute similarity between samples
def sim_matrix(samples, percept):
    sim = np.zeros((len(samples), len(samples)))

    for i, x1 in enumerate(samples.values()):
        x1 = torch.from_numpy(x1).permute(2,0,1).unsqueeze(0).float().to('cuda') / 255
        for j, x2 in enumerate(samples.values()):
            #sim[i,j] = jpg_sim(x1, x2)
            x2 = torch.from_numpy(x2).permute(2,0,1).unsqueeze(0).float().to('cuda') / 255
            sim[i,j] = percept(x1, x2).sum().item()

    return sim

import gzip
from einops import rearrange
def gzip_sim(x1, x2):
    #x = np.concatenate((x1, x2), axis=0)
    x = np.stack((x1, x2), axis=0)
    x = rearrange(x, 's c h w -> c h w s')

    # get size of compressed data
    c1 = gzip.compress(x1.tobytes())
    c2 = gzip.compress(x2.tobytes())
    c = gzip.compress(x.tobytes())
    return 100 * len(c) / (len(c1) + len(c2)) / 2

import cv2
def jpg_sim(x1, x2):
    x = np.concatenate((x1, x2), axis=0)

    enc = [int(cv2.IMWRITE_PNG_COMPRESSION), 5]
    c1 = len(cv2.imencode('.png', x1, enc)[1])
    c2 = len(cv2.imencode('.png', x2, enc)[1])
    c = len(cv2.imencode('.png', x, enc)[1])

    return 2*c / (c1 + c2)

if __name__ == '__main__':
    data_path = '/drive2/ood/'
    datasets = ['svhn', 'svhn', 'cifar10', 'cifar10']

    # get loaders
    loaders = {}
    for i, dataset in enumerate(datasets):
        loaders[dataset + ' - ' + str(i)] = get_loader(data_path, dataset, 'train', 1, 8)

    # perceptual similarity
    percept = Percept().eval().to('cuda')

    while True:
        # get one sample from each
        samples = sample_one(loaders)

        # compute similarity
        sim = sim_matrix(samples, percept)

        # plot
        n = len(datasets)
        fig, axs = plt.subplots(1, n + 1, figsize=(n*4, 4))
        for i, (dataset, img) in enumerate(samples.items()):
            axs[i].imshow(img)
            axs[i].set_title(dataset)

        # label ticks of similarity matrix to match datasets in samples
        axs[n].imshow(sim, cmap='gray', vmin=1.5*np.min(sim), vmax=np.max(sim))
        for i, dataset in enumerate(datasets):
            axs[n].text(i, i, dataset, ha='center', va='center')

        # plot values in matrix (except diagonal)
        for i in range(len(datasets)):
            for j in range(len(datasets)):
                if i != j:
                    axs[n].text(j, i, '{:.3f}'.format(sim[i,j]), ha='center', va='center')

        plt.tight_layout()
        plt.show()
        