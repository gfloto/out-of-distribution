import os 
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from dist_matrix import DistTune, z_dist
from models.autoenc import get_autoenc
from datasets import all_datasets, get_loader
from sklearn.metrics import roc_curve, auc

'''
the following functions are for generating data given
trained autoencder model. this is expected to perform
worse than directly optimizing for latents...


note: this shouldn't be used at this point...
instead, we can just save both the initial and tuned state as [2, 1024] tensors
'''

# save latent space representations of all datasets
@torch.no_grad()
def z_store(model, loader, mode, device):
    z_all = None; recon_all = None
    for i, (x, _) in enumerate(tqdm(loader)):
        if i == 4 and mode == 'test': break
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

# given experiment name, save latent space representations of all datasets
def save_autoenc_latents(train_args, device, batch_size=2048):
    # make dir of compressed representations
    save_path = os.path.join('results', train_args.test_name, 'autoenc_lat')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load model
    model = get_autoenc(train_args).to(device)
    model.load_state_dict(torch.load(os.path.join('results', train_args.test_name, 'model.pt')))

    datasets = all_datasets()
    datasets = ['cifar10']

    for i, dataset in enumerate(datasets):
        if dataset == train_args.dataset: 
            modes = ['train', 'test']
        else: 
            modes = ['test']

        for mode in modes:
            loader = get_loader(train_args.data_path, dataset, mode, batch_size)
            z, recon = z_store(model, loader, mode, device)

            # save z
            torch.save(z, os.path.join(save_path, f'{dataset}_{mode}_z.pt'))

'''
the following functions are for optimizing latents
directly, relative to the training latents. For now,
it will be assumed that the training latents are
available, via the save_latents function below.
'''

def save_tuned_latents(train_args, device, batch_size=64):
    # make dir of compressed representations
    save_path = os.path.join('results', train_args.test_name, 'tuned_lat')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = all_datasets()
    datasets.remove('cifar10')
    #datasets = datasets[5:]
    datasets = ['cifar10'] + datasets

    # get train dataloader
    train_loader = get_loader(train_args.data_path, train_args.dataset, 'train', batch_size)

    # load model
    model = get_autoenc(train_args).to(device)
    model.load_state_dict(torch.load(os.path.join('results', train_args.test_name, 'model.pt')))

    # get lpips distance object
    dist_tune = DistTune('inner_prod', 'cuda')

    for i, dataset in enumerate(datasets):
        test_loader = get_loader(train_args.data_path, dataset, 'test', batch_size=1)
        loss = z_tune(train_loader, test_loader, model, dist_tune, device)
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

# tune latent representations to respect lpips distance to training data...
def z_tune(train_loader, test_loader, model, dist_tune, device):
    z1_test_all = []; z2_test_all = []
    model.eval()

    # get a batch of test data to tune
    #for i, (x_test, _) in enumerate(tqdm(test_loader)):
    total_loss = []
    for i, (x_test, _) in enumerate(test_loader):
        if i == 100: break
        x_test = x_test.to(device)

        # latent representation of test data
        z_test = model(x_test)[1].detach().clone() 
        z1_test_all.append(z_test[0].clone()) # save initial latent

        # make latents a parameter to optimize
        z_test = torch.nn.Parameter(z_test)
        optim = torch.optim.Adam([z_test], lr=0.2)

        # get batches of train data
        lat_loss = 0
        for j, (x_train, _) in enumerate(train_loader):
            if j == 20: break
            x_train = x_train.to(device)

            # ideal distance from lpips
            dist = dist_tune(x_train, x_test)

            # get inner product distance between latents
            _, z_train, _ = model(x_train)
            lat_dist = (z_train * z_test).sum(dim=1)
            
            # loss
            loss = (dist - lat_dist).square().mean()

            # optimize latent
            optim.zero_grad()
            loss.backward()
            optim.step()

            # normalize embeddings to be on unit sphere
            z_test.data /= z_test.data.norm(dim=1, keepdim=True)

        # save tuned latent
        z2_test_all.append(z_test[0].clone())

        # get loss between: x_test and model.decode(z2_test)
        x_out = model.decode(z_test)
        recon_loss = (x_test - x_out).square().mean()

        # get loss based on how far z2_test is from z1_test
        lat_loss = (z1_test_all[-1] - z2_test_all[-1]).square().mean()

        #loss = recon_loss + lat_loss
        loss = lat_loss
        total_loss.append(loss.item())

        # total loss
        #print(f'recon_loss: {recon_loss.item():.4f}, lat_loss: {lat_loss.item():.4f}, total_loss: {loss.item():.4f}')
    print()
    return total_loss

if __name__ == '__main__':
    test_name = 'lat_36'
    mode = 'tuned'
    assert mode in ['tuned', 'autoenc']

    # load autoencoder model
    with open(os.path.join('results', test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)
    save_path = os.path.join('results', test_name, 'diffusion')
    os.makedirs(save_path, exist_ok=True)

    if mode == 'autoenc':
        print(f'generating latents with autoencoder from {test_name}')

        # save latents from autoencoder
        save_autoenc_latents(train_args, 'cuda')

    elif mode == 'tuned':
        print(f'generating latents via tuning from {test_name}')
        save_tuned_latents(train_args, 'cuda')

