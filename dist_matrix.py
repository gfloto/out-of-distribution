import sys, os
import math
import torch
import numpy as np
from tqdm import tqdm

from percept import Percept
from datasets import get_loader

# pairwise distance matrix using perceptual distance
class DistMatrix:
    def __init__(self, loader, args):
        self.device = args.device
        self.metric = args.metric
        self.model = Percept().eval().to(args.device)

        self.mean = None; self.std = None
        self.mean, self.std = self.get_mean_std(loader, args.device)
        #self.mean = 0.3062; self.std = 0.2787
        print(f'using mean: {self.mean:.5f} and std: {self.std:.5f} for normalizing perceptual distance matrix')

    # get mean and std of perceptual distance to center data
    def get_mean_std(self, loader, device):
        value = []
        for i, (x, _) in enumerate(tqdm(loader)):
            if i > 100: break

            x = x.to(device)
            dist = self.__call__(x)

            # get flattened upper triangular values
            for i in range(x.shape[0]):
                for j in range(i+1, x.shape[0]):
                    value.append(dist[i,j].item())
        value = np.array(value)
        value = np.log(value) - np.log(1 - value)

        # convert to tensor
        value = torch.tensor(value).to(device)
        return value.mean(), value.std()
        
    @ torch.no_grad()
    def __call__(self, x, batch_size=1024):
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
            if self.mean is not None:
                dist_batch = dist_batch.log() - (1 - dist_batch).log()
                dist_batch = (dist_batch - self.mean) / self.std
                dist_batch = torch.sigmoid(dist_batch)

                # l2 has close 0, far 2 (z on unit sphere)
                # inner prod has close 1, far -1
                if self.metric == 'inner_prod':
                    dist_batch = -2*dist_batch + 1
                elif self.metric == 'l2':
                    dist_batch = 2*dist_batch

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
def z_dist(x, metric):
    assert len(x.shape) == 2

    g = x @ x.T
    if metric == 'inner_prod':
        return g
    elif metric == 'l2':
        s = g.shape[0]
        a = g.diag().repeat(s, 1)
        return a.T + a - 2*g
    else: raise ValueError('invalid distance mode')

# ensure that dist matrix is correct
def test_dist(data_path):
    # test torch cdist
    a = torch.randn(64, 12)
    dist = z_dist(a, 'l2')

    # check correctness
    for i in range(100):
        i = np.random.randint(0, a.shape[0])
        j = np.random.randint(0, a.shape[0])

        assert torch.allclose(dist[i,j], (a[i] - a[j]).norm())
    print('\nlatent distance matrix test passed\n')
    quit()

    # -------------------

    # loader    
    loader = get_loader(data_path, 'cifar10', 'train', 32)
    dist_matrix = DistMatrix(loader, 'cuda')

    # get sample
    x, _ = next(iter(loader))
    x = x.to('cuda')

    # check distance matrix
    dist = dist_matrix(x)

    # check for 100 random samples that dist is correct
    percept = dist_matrix.model
    for i in range(100):
        i = np.random.randint(0, x.shape[0])
        j = np.random.randint(0, x.shape[0])

        assert dist[i,j] - percept(x[i], x[j]) < 1e-4
    print('\nperceptual distance matrix test passed\n')

if __name__ == '__main__':
    data_path = '/drive2/ood/'
    test_dist(data_path)
    quit()

    # check distribution of distances
    loader = get_loader(data_path, 'cifar10', 'train', 32, 4)
    dist_matrix = DistMatrix('cuda')

    value = []
    for n, (x, _) in enumerate(loader):
        if n > 100: break

        x = x.to('cuda')
        dist = dist_matrix(x)

        # get flattened upper triangular values
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                value.append(dist[i,j].item())

    value = np.array(value)
    value = np.log(value) - np.log(1 - value)
    print(value.mean(), value.std())
    value = (value - value.mean()) / value.std()

    # plot histogram compared to normal distribution
    import matplotlib.pyplot as plt
    plt.hist(value, bins=50, density=True, alpha=0.5)
    plt.hist(np.random.randn(1000), bins=50, density=True, alpha=0.5)
    plt.show()




        
