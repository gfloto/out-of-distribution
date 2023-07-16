import os, sys
import json
import torch
import argparse
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn import metrics
from einops import rearrange

from test import test
from loss import Loss
from models import get_model
from dataloader import get_dataloader

# get args for testing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='damage-2', help='path to model checkpoint')
    parser.add_argument('--save_name', type=str, default='model_sup_tune.pt', help='name of test folder')
    parser.add_argument('--tune_type', type=str, default='sup', help='sup or unsup')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--early_stop', type=int, default=10, help='early stopping condition')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='device type')

    args = parser.parse_args()

    # update paths
    args.model_load_path = os.path.join('results', args.test_name, 'model.pt')
    args.model_save_path = os.path.join('results', args.test_name, args.save_name)

    # check inputs
    if not os.path.exists(args.model_load_path):
        raise ValueError('model_path does not exist')
    if os.path.exists(args.model_save_path):
        raise ValueError('tune_name already exists')

    return args

# get mu len
def get_mu_len(mu, spheres=1):
    if spheres == 1:
        return mu.linalg.norm(dim=-1).mean().item()
    else:
        mu = rearrange(mu, 'b (s d) -> b s d', s=spheres)
        return torch.linalg.norm(mu, dim=-1).mean().item()

# tuning loop in unsupervised setting
def unsup_tune(model, loaders, loss_fns, device='cuda'):
    model.train()
    spheres = loss_fns['in'].spheres  
    opt = optim.Adam(model.parameters(), lr=1e-5) 

    flip = True
    train_track, test_track, ood_track = [], [], [] # tracking avg distance from origin
    # backprop samples before test
    for i in tqdm(range(200)):   
        # get samples, TODO: this is slow
        if flip:
            mode = 'train'
            x, label = next(iter(loaders['train']))
        else:
            mode = 'test'
            x, label = next(iter(loaders['test']))
        flip = not flip 

        # prep info
        x = x.to(device)
        angle = label['angle'].to(device)
        reason = label['reason'] 

        # update params
        mu, _, x_out = model(x, angle)
        if mode == 'train':
            recon, percept, kld = loss_fns['in'](x, x_out, mu)
            loss = recon + percept + kld
        else:
            recon, percept, kld = loss_fns['out'](x, x_out, mu)
            loss = recon + percept + 0.4*kld

        # optimize
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        
        # tracking distance of mu from origin
        if mode == 'train':
            train_track.append(get_mu_len(mu, spheres))
        else:
            id_mu = mu[torch.where(reason == 'Accepted')[0]]
            ood_mu = mu[torch.where(reason != 'Accepted')[0]]

            test_track.append(get_mu_len(id_mu, spheres))
            ood_track.append(get_mu_len(ood_mu, spheres))

    # print avg distance from origin
    print('train: ', np.mean(train_track))
    print('test: ', np.mean(test_track))
    print('ood: ', np.mean(ood_track))

    return model

# tuning loop in supervised setting
def sup_tune(model, loaders, loss_fns, device='cuda'):
    model.train()
    spheres = loss_fns['in'].spheres
    opt = optim.Adam(model.parameters(), lr=1e-4) 

    # backprop samples before test
    for i, (x, label) in enumerate(tqdm(loaders['mixed_train'])):   
        if i > 1000: break
        # prep info
        x = x.to(device)
        angle = label['angle'].to(device)
        reason = label['reason'] 

        # update params
        mu, _, x_out = model(x, angle)

        # sort by id and ood
        reason = np.array(reason)
        id_idx = np.where(reason == 'Accepted')
        ood_idx = np.where(reason != 'Accepted')

        id_recon, id_percept, id_kld = loss_fns['in'](x[id_idx], x_out[id_idx], mu[id_idx])
        ood_recon, ood_percept, ood_kld = loss_fns['out'](x[ood_idx], x_out[ood_idx], mu[ood_idx])
        loss = id_recon + id_percept + id_kld + ood_recon + ood_percept + ood_kld

        # optimize
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()

    return model


if __name__== "__main__":
    tune_args = get_args()

    # load .json from test_name
    with open(os.path.join('results', tune_args.test_name, 'args.json'), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # load datasets
    if tune_args.tune_type == 'sup':
        # typical train/eval split
        tune = sup_tune
        loader_types = ['mixed_train', 'eval']
    elif tune_args.tune_type == 'unsup':
        # weird split used in the vae tilt paper
        tune = unsup_tune
        loader_types = ['train', 'test', 'eval']
    loaders = [get_dataloader(args, mode=type) for type in loader_types]
    loaders = dict(zip(loader_types, loaders))

    # loss functions
    loss_types = ['in', 'out']
    loss_fns = (Loss(args, args.tilt), Loss(args, None))
    loss_fns = dict(zip(loss_types, loss_fns))

    # make model, get state dict
    model = get_model(args.model_yaml_path, args.lat_dim, args.use_timestep).to(args.device)
    load_path = os.path.join('results', args.test_name, 'model.pt')
    state = torch.load(load_path, map_location=args.device)

    # load original params
    model.load_state_dict(state)

    # tune loops on new data
    auc_track = []
    for i in range(1000):
        model = tune(model, loaders, loss_fns, args.device)
        scores, reason_ids = test(model, loaders['eval'], loss_fns['in'], args.device, tune=True)

        # get final scores and results
        labels = reason_ids == 'Accepted'
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        aucroc = metrics.auc(fpr, tpr)

        # track aurcroc
        auc_track.append(aucroc)
        print('aucroc: {:.5f}'.format(aucroc))

        # early stopping
        if len(auc_track) > tune_args.early_stop: 
            auc_track.pop(0)
        if np.argmax(auc_track) == 0 and i > tune_args.early_stop: 
            print('early stopping')
            break

        # save tuned model based on best aucroc
        if aucroc == np.max(auc_track):
            # save model
            torch.save(model.state_dict(), tune_args.model_save_path)
            print('saved model')
