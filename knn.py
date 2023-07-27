import os 
import torch 
import json
from tqdm import tqdm
from argparse import Namespace

import numpy as np
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt

from args import get_args
from models import get_model
from dist_matrix import z_dist
from datasets import get_loader, all_datasets

@torch.no_grad()
def z_store(model, loader, device):
    z_all = None; recon_all = None
    for i, (x, _) in enumerate(tqdm(loader)):
        if i == 4: break
        x = x.to(device)

        _, mu, x_out = model(x)
        recon = (x - x_out).square().mean(dim=(1,2,3))

        if z_all is None:
            z_all = mu
            recon_all = recon
        else:
            z_all = torch.cat((z_all, mu), dim=0)
            recon_all = torch.cat((recon_all, recon), dim=0)
    
    return z_all, recon_all

def save_z(name, device):
    batch_size = 2048

    # make dir of compressed representations
    save_path = os.path.join(name, 'lat')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get args
    with open(os.path.join(name, 'args.json'), 'r') as f:
        args = json.load(f)
        args = Namespace(**args)

    # load model
    args.lat_dim = 256
    model = get_model(args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(name, 'model.pt')))

    datasets = all_datasets()
    for i, dataset in enumerate(datasets):
        print(f'dataset: {dataset}')
        loader = get_loader(args.data_path, dataset, 'train', batch_size)

        z, recon = z_store(model, loader, device)

        # save z
        torch.save(z, os.path.join(save_path, f'{dataset}_z.pt'))
        torch.save(recon, os.path.join(save_path, f'{dataset}_recon.pt'))
        print(z.shape)

def get_data(dataset):
    z = torch.load(os.path.join(name, 'lat', f'{dataset}_z.pt')).to(device)
    recon = torch.load(os.path.join(name, 'lat', f'{dataset}_recon.pt')).cpu().numpy()

    return z, recon

def get_nn(z_train, z_test=None, k=1): 
    if z_test is not None:
        # compute distance matrix
        mix = torch.cat((z_train, z_test), dim=0)
        d = z_dist(mix, 'l2')

        s = d.shape[0]
        d[torch.arange(s), torch.arange(s)] = float('inf')
        d = d[:s//2, s//2:]
    else:
        d = z_dist(z_train, 'l2')
        s = d.shape[0]
        d[torch.arange(s), torch.arange(s)] = float('inf')

    # sort each row of distance matrix
    nn_arg = torch.argsort(d, dim=1)[:, :k]

    # get nearest neighbor
    s = d.shape[0]
    nn = torch.empty((s,))
    for i in range(s):
        nn[i] = d[i, nn_arg[i]].mean()

    return nn.cpu().numpy()

if __name__ == '__main__':
    k = 1
    name = 'results/l2_lat256'
    device = 'cuda'
    #save_z(name, device)

    train_dataset = 'cifar10'
    datasets = all_datasets()
    datasets.remove(train_dataset)

    # get train data
    z_train, recon_train = get_data(train_dataset)
    nn_train = get_nn(z_train, k=k)

    # plotting 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, dataset in enumerate(datasets):
        print(f'dataset: {dataset}')

        # get test data
        z_test, recon_test = get_data(dataset)
        nn_test = get_nn(z_train, z_test, k=k) 

        # calculate aucroc
        train_score = nn_train + recon_train
        test_score = nn_test + recon_test

        y_true = np.concatenate((np.zeros_like(train_score), np.ones_like(test_score)))
        y_score = np.concatenate((train_score, test_score))
        aucroc = roc_auc_score(y_true, y_score)
        print(f'aucroc: {aucroc:.5f}\n')

        # plot with matplotlib
        ax.hist(nn_test, bins=100, alpha=0.5, label=dataset, density=True)

    ax.hist(nn_train, bins=100, alpha=0.5, label=train_dataset, density=True)
    plt.legend()
    plt.show()
