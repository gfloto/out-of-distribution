import sys, os 
import json
import torch
from tqdm import tqdm
from argparse import Namespace

from utils import ptnp
from loss import Loss
from models import get_model
from metrics import metrics
from datasets import all_datasets, get_loader

@torch.no_grad()
def test(model, loader, loss_fn, device, consistent_score=False):
    model.eval()
    
    for i, (x, _) in enumerate(loader):            
        if i > 25: break
        x = x.to(device)

        # push through model
        if not consistent_score:
            _, mu, x_out = model(x, test=True)
            recon, percept, kld = loss_fn(x, x_out, mu, test=True)
            score = recon + percept + kld
        else:
            _, mu_0, x_out = model(x)
            _, mu_1, _ = model(x_out)
            score = (mu_0 - mu_1).pow(2).sum(dim=-1)

        if i == 0:
            score_track = score
        else:
            score_track = torch.cat((score_track, score), dim=0)
    
    return ptnp(score_track)

# run ood vs id testing on all available datasets
def ood_test(model, loss_fn, args, consistent_score=False):
    dataset_names = all_datasets()

    score_track = {}
    for dataset in dataset_names:
        loader = get_loader(args.data_path, dataset, 'test', args.batch_size, args.workers)
        score = test(model, loader, loss_fn, args.device, consistent_score=consistent_score)
        score = test(model, loader, loss_fn, args.device) 
        score_track[dataset] = score

    return score_track

if __name__ == '__main__':
    path = 'results/pre'
    device = 'cuda'

    # load training args
    with open(os.path.join(path, 'args.json'), 'r') as f:
        args = json.load(f)
    args = Namespace(**args)
    args.device = device # TODO: fix this
    train_dataset = args.dataset

    # load model
    model = get_model(args.lat_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
    loss_fn = Loss(args)

    # nll score
    save_path = os.path.join(path, 'metrics_nll.json')
    score_track = ood_test(model, loss_fn, args)
    nll = metrics(score_track, train_dataset)
    with open(save_path, 'w') as f:
        json.dump(nll, f)

    # consistent score
    save_path = os.path.join(path, 'metrics_consistent.json')
    score_track = ood_test(model, loss_fn, args, consistent_score=True)
    consist = metrics(score_track, train_dataset)
    with open(save_path, 'w') as f:
        json.dump(consist, f)
