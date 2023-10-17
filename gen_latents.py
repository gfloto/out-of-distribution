import os 
import json
import torch
import argparse
from tqdm import tqdm

from dist_matrix import DistTune, z_dist
from models.autoenc import get_autoenc
from datasets import all_datasets, get_loader

'''
the following functions are for generating data given
trained autoencder model. this is expected to perform
worse than directly optimizing for latents...
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
    save_path = os.path.join('results', train_args.test_name, 'diffusion', 'autoenc_lat')
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
            #torch.save(recon, os.path.join(save_path, f'{dataset}_{mode}_recon.pt'))

'''
the following functions are for optimizing latents
directly, relative to the training latents. For now,
it will be assumed that the training latents are
available, via the save_latents function below.
'''

def save_tuned_latents(train_args, device, batch_size=32):
    # make dir of compressed representations
    save_path = os.path.join('results', train_args.test_name, 'diffusion', 'tuned_lat')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = all_datasets()

    # get train dataloader
    train_loader = get_loader(train_args.data_path, train_args.dataset, 'train', batch_size)

    # load model
    model = get_autoenc(train_args).to(device)
    model.load_state_dict(torch.load(os.path.join('results', train_args.test_name, 'model.pt')))

    # get lpips distance object
    dist_tune = DistTune('inner_prod', 'cuda')

    for i, dataset in enumerate(datasets):
        print(dataset)
        test_loader = get_loader(train_args.data_path, dataset, 'test', batch_size=1)
        z, z_loss = z_tune(train_loader, test_loader, model, dist_tune, device)
        print(z_loss.mean(), z_loss.std())

        # save z
        torch.save(z, os.path.join(save_path, f'{dataset}_z.pt'))
        torch.save(z_loss, os.path.join(save_path, f'{dataset}_loss.pt'))

@ torch.no_grad()
def compute_loss(train_loader, model, dist_tune, y_lat, device):
    x = train_loader.dataset.x[-1024:].to(device)
    
    dist = dist_tune(x, y) 
    _, x_lat, _ = model(x)
    


def z_tune(train_loader, test_loader, model, dist_tune, device):
    y_lat_all = []; y_loss_all = []
    model.eval()

    for i, (y, _) in enumerate(tqdm(test_loader)):
        if i == 1000: break

        # with gradient for y
        y_lat = model(y.to(device))[1].detach().clone() 
        y_lat = torch.nn.Parameter(y_lat)

        optim = torch.optim.Adam([y_lat], lr=0.25)

        for j, (x, _) in enumerate(train_loader):
            if j == 12: break
            x = x.to(device)
            y = y.to(device)

            # ideal distance from lpips
            dist = dist_tune(x, y)

            # get inner product distance
            _, x_lat, _ = model(x)
            lat_dist = (x_lat * y_lat).sum(dim=1)
            
            # loss
            loss = (dist - lat_dist).square().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

            # normalize y_lat
            y_lat.data /= y_lat.data.norm(dim=1, keepdim=True)

        # compute loss on large batch
        loss = compute_loss(train_loader, model, dist_tune, y_lat, device)

        # save y_lat
        y_lat_all.append(y_lat.data)
        y_loss_all.append(loss.item())

    return torch.cat(y_lat_all, dim=0), torch.tensor(y_loss_all) 

if __name__ == '__main__':
    test_name = 'lat_36'
    mode = 'autoenc'
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

