import os 
import json
import torch
import argparse
import numpy as np
from einops import rearrange

from dist_matrix import DistTune
from models.autoenc import get_autoenc
from datasets import all_datasets, get_loader
from sklearn.metrics import roc_curve, auc

'''
the following functions are for optimizing latents
directly, relative to the training latents
'''

def tune_latents(train_args, device, batch_size=2048):
    datasets = all_datasets()
    datasets.remove(train_args.dataset)
    datasets = datasets[5:]
    datasets = [train_args.dataset] + datasets

    # get train dataloader
    train_loader = get_loader(train_args.data_path, train_args.dataset, 'train', batch_size)

    # load model
    model = get_autoenc(train_args).to(device)
    model.load_state_dict(torch.load(os.path.join('results', train_args.test_name, 'model.pt')))

    # get lpips distance object
    dist_tune = DistTune(train_args.lpips_mean, train_args.lpips_std, device)

    for i, dataset in enumerate(datasets):
        print(f'tuning latents for {dataset}')
        test_loader = get_loader(train_args.data_path, dataset, 'test', batch_size=1)
        loss = z_tune(
            train_loader, test_loader, model, dist_tune, 
            batch_size, train_args, dataset, device
        )
        loss = np.array(loss)
        
        if i == 0: mean = loss.mean(axis=0)
        loss /= mean
        loss = loss.sum(axis=1)

        if i == 0:
            id_scores = loss
            id_labels = np.zeros_like(id_scores)
        else:
            ood_scores = loss
            ood_labels = np.ones_like(ood_scores)

            scores = np.concatenate((id_scores, ood_scores), axis=0)
            labels = np.concatenate((id_labels, ood_labels), axis=0)
            fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
            aucroc = auc(fpr, tpr)

            print(f'{dataset} aucroc: {aucroc:.4f}')

@torch.no_grad()
def get_z(x, model):
    # encoder outputs (z, mu)
    z = model.encode(x)[1].detach().clone()
    z = model.sphere_norm(z)
    return z

@torch.no_grad()
def get_recon(z, model):
    z = rearrange(z, 'b s d -> b (s d)')
    x_out = model.decode(z)
    return x_out

import time
import math

from plot import save_tune_sample

# tune latent representations to respect lpips distance to training data...
def z_tune(train_loader, test_loader, model, dist_tune, batch_size, args, dataset, device):
    model.eval()
    spheres = args.spheres
    lat_dim = args.lat_dim

    # get a test image to perform ood on
    total_loss = []
    for i, (x_test, _) in enumerate(test_loader):
        if i == 100: break
        x_test = x_test.to(device)

        # latent representation of test data
        z_test = get_z(x_test, model)
        z_test = rearrange(z_test, 'b (s d) -> b s d', s=spheres, d=lat_dim)
        z_init = z_test.clone()

        # make latent a parameter to optimize
        z_test = torch.nn.Parameter(z_test)
        optim = torch.optim.Adam([z_test], lr=0.1)

        # get test features, make batch_size copies
        feat_test = dist_tune.feats(x_test)
        for j in range(len(feat_test)):
            feat_test[j] = feat_test[j].repeat(batch_size, 1, 1, 1)

        # get large batch of train data
        x_train = next(iter(train_loader))[0].to(device)
        feat_train = dist_tune.feats(x_train)
        z_train = get_z(x_train, model)

        # compute distance between test and train features
        dist = dist_tune(feat_train, feat_test)

        # backprop
        for j in range(32):
            bs = 64
            z_tr = z_train[j*bs:(j+1)*bs]
            di = dist[j*bs:(j+1)*bs]

            # get latent distance
            z_tr = rearrange(z_tr, 'b (s d) -> b s d', s=spheres, d=lat_dim)
            lat_dist = torch.einsum('aij , bij -> ab', z_test, z_tr)[0]
            
            # loss
            loss = (di - lat_dist).square().mean()

            # optimize latent
            optim.zero_grad()
            loss.backward()
            optim.step()

            # normalize embeddings to be on unit sphere
            z_test = rearrange(z_test, 'b s d -> (b s) d')
            z_test.data /= z_test.data.norm(dim=-1, keepdim=True)
            z_test = rearrange(z_test, '(b s) d -> b s d', s=spheres, d=lat_dim)
            z_test.data = z_test.data / math.sqrt(spheres)

            if j == 0: loss_init = loss.detach().item()

        # get reconstruction performance
        x_init = get_recon(z_init, model)
        x_tuned = get_recon(z_test, model)

        # save samples
        save_path = os.path.join('results', args.test_name, 'tune', f'{dataset}_{i}.png')
        save_tune_sample(x_test, x_init, x_tuned, save_path)

        recon_init = dist_tune(x_init, x_test, feats=False).mean().item()
        recon_init = (1 - recon_init) / 2
        #recon_tuned = dist_tune(x_tuned, x_test, feats=False)
        #recon_loss = (recon_tuned - recon_init).square().mean().item()

        recon_loss = dist_tune(x_init, x_tuned, feats=False).mean().item()
        recon_loss = (1 - recon_loss) / 2

        # get loss based on how far z2_test is from z1_test
        loss_lat = (z_init - z_test).square().mean().item()
        total_loss.append([loss_lat, recon_loss, loss_init, recon_init])

        # total loss
        #print(f'lat loss: {loss_lat:.4f} init loss: {loss_init:.4f} recon loss: {recon_loss:.4f}')
        print(f'lat loss: {loss_lat:.4f} recon loss: {recon_loss:.4f} init lat: {loss_init:.4f} init recon: {recon_init:.4f}')
    return total_loss

if __name__ == '__main__':
    device = 'cuda'
    test_name = 'cifar10-prodsphere2'

    # load autoencoder model
    with open(os.path.join('results', test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)

    tune_latents(train_args, device)

