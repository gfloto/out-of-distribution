import os 
import torch 
import json
from argparse import Namespace

import numpy as np
from einops import rearrange
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt

from models.autoenc import get_autoenc
from dist_matrix import z_dist
from datasets import all_datasets

def get_data(dataset, mode):
    z = torch.load(os.path.join(name, 'lat', f'{dataset}_{mode}_z.pt')).to(device)
    recon = torch.load(os.path.join(name, 'lat', f'{dataset}_{mode}_recon.pt')).cpu().numpy()

    return z[:4096], recon[:4096]

def get_nn(z_train, z_test=None, k=1): 
    if z_test is not None:
        # compute distance matrix
        mix = torch.cat((z_train, z_test), dim=0)
        d = z_dist(mix, 'l2')

        # only look at relation between z_train and z_test
        s = z_train.shape[0]
        d[torch.arange(s), torch.arange(s)] = float('inf')
        d = d[:s, s:]

        # checking...
        for _ in range(100):
            i = np.random.randint(z_train.shape[0])
            j = np.random.randint(z_test.shape[0])
            assert torch.allclose(d[i,j], (z_train[i] - z_test[j]).square().sum(), atol=1e-4)
    else:
        d = z_dist(z_train, 'l2')
        s = d.shape[0]
        d[torch.arange(s), torch.arange(s)] = float('inf')

        # checking...
        for _ in range(100):
            i = np.random.randint(z_train.shape[0])
            j = np.random.randint(z_train.shape[0])
            if i == j: assert d[i,j] == float('inf')
            assert torch.allclose(d[i,j], (z_train[i] - z_train[j]).square().sum(), atol=1e-4)

    # get k nearest distances from each test point (col) to any train point (row)
    nn_arg = torch.argsort(d, dim=1)[:, :k]

    # get nearest neighbor
    s = d.shape[1]
    nn = torch.empty((s,))
    for i in range(s):
        nn[i] = d[nn_arg[i], i].mean()

    return nn.cpu().numpy()

@ torch.no_grad()
def vis(model, z, dataset):
    k = 16
    # visualize decoded z_train
    x = model.decode(z[:k])
    x = x.clamp(0, 1)

    # rearrange into 4x4 grid
    x = rearrange(x, '(b1 b2) c h w -> b1 b2 c h w', b1=4, b2=4)
    x = rearrange(x, 'b1 b2 c h w -> (b1 h) (b2 w) c')

    # plot
    plt.imshow(x.cpu().numpy())
    plt.title(dataset)
    plt.show()

if __name__ == '__main__':
    k = 300
    alpha = 2*k
    vis = False
    name = 'results/l2_cifar10_256'
    train_dataset = 'cifar10'
    device = 'cuda'
    #save_z(name, train_dataset, device)

    # get dataset info
    datasets = all_datasets()
    datasets.remove(train_dataset)

    # ---------------------

    # get args
    with open(os.path.join(name, 'args.json'), 'r') as f:
        args = json.load(f)
        args = Namespace(**args)

    # load model
    args.lat_dim = 256
    model = get_autoenc(args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(name, 'model.pt')))

    # ---------------------

    # get train data
    z_train, recon_train = get_data(train_dataset, 'train')
    z_test, recon_test = get_data(train_dataset, 'test')
    if vis: vis(model, z_train, train_dataset)

    nn_train = get_nn(z_train, z_test, k=k)
    train_score = nn_train + alpha*recon_train

    # plotting 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, dataset in enumerate(datasets):
        print(f'dataset: {dataset}')

        # get test data
        z_test, recon_test = get_data(dataset, 'test')
        if vis: vis(model, z_test, dataset)
        nn_test = get_nn(z_train, z_test, k=k)

        #nn_test = []
        #for j in tqdm(range(z_test.shape[0])):
            #z_t = z_test[j].unsqueeze(0)
            #nn_t = get_nn(z_train, z_t, k=k) 
            #nn_test.append(nn_t[0])
        #nn_test = np.array(nn_test)

        # calculate aucroc
        test_score = nn_test + alpha*recon_test

        y_true = np.concatenate((np.zeros_like(train_score), np.ones_like(test_score)))
        y_score = np.concatenate((train_score, test_score))
        aucroc = roc_auc_score(y_true, y_score)
        print(f'aucroc: {aucroc:.5f}\n')

        # plot with matplotlib
        ax.hist(nn_test, bins=100, alpha=0.5, label=dataset, density=True)

    ax.hist(nn_train, bins=100, alpha=0.5, label=train_dataset, density=True)
    plt.legend()
    plt.show()
