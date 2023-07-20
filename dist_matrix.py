import sys, os
import math
import torch
import numpy as np
from einops import rearrange

from percept import Percept
from datasets import get_loader

# pairwise distance matrix using perceptual distance
class DistMatrix:
    def __init__(self, device):
        self.model = Percept().eval().to(device)
        
    @ torch.no_grad()
    def __call__(self, x, batch_size=4096):
        # rearrange x into batches to compute pairwise distances
        # we only need to compute the upper triangular matrix
        self.x1_all = []; self.x2_all = []
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                self.x1_all.append(x[i])
                self.x2_all.append(x[j])

        # compute pairwise distances in batches
        dist = None
        n = math.ceil(len(self.x1_all) / batch_size)
        for i in range(n):
            # stack into batches
            x1 = torch.stack(self.x1_all[i*batch_size:(i+1)*batch_size])
            x2 = torch.stack(self.x2_all[i*batch_size:(i+1)*batch_size])

            # compute and store distances
            dist_batch = self.model(x1, x2)
            dist = torch.cat((dist, dist_batch)) if dist is not None else dist_batch 

        # convert to upper triangular matrix
        c = 0
        out = torch.zeros((x.shape[0], x.shape[0])).to(x.device)
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                out[i,j] = dist[c]
                c += 1

        return out + out.T + torch.eye(x.shape[0]).to(x.device)

# pairwise gram matrix
def z_dist(x1, x2):
    assert x1.shape == x2.shape
    assert len(x1.shape) == 2

    g = x1 @ x2.T
    return g

if __name__ == '__main__':
    data_path = '/drive2/ood/'

    # test torch cdist
    a = torch.randn(64, 12)
    b = torch.randn(64, 12)
    dist = z_dist(a, b)

    # check correctness
    for i in range(100):
        i = np.random.randint(0, a.shape[0])
        j = np.random.randint(0, a.shape[0])

        assert dist[i,j] - (a[i] @ b[j]) < 1e-5
    print('\nlatent distance matrix test passed\n')

    # -------------------

    # loader    
    loader = get_loader(data_path, 'cifar10', 'train', 32, 4)
    dist_matrix = DistMatrix('cuda')

    # get sample
    x, _ = next(iter(loader))
    x = x.to('cuda')

    # check distance matrix
    dist = dist_matrix(x)
    print(dist)

    # check for 100 random samples that dist is correct
    percept = dist_matrix.model
    for i in range(100):
        i = np.random.randint(0, x.shape[0])
        j = np.random.randint(0, x.shape[0])

        assert dist[i,j] - percept(x[i], x[j]) < 1e-4
    print('\nperceptual distance matrix test passed\n')
        
