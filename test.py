import os 
import json
import torch
from argparse import Namespace

from utils import ptnp
from loss import Loss
from models.autoenc import get_autoenc
from metrics import metrics
from datasets import all_datasets, get_loader

@torch.no_grad()
def test(model, train_loader, test_loader, loss_fn, device):
    model.eval()

    # iterate through both dataloaders
    for i, ((x_train, _), (x_test, _)) in enumerate(zip(train_loader, test_loader)): 
        if x_train.shape != x_test.shape: break
    #for i, (x_test, _) in enumerate(test_loader):
        #if i > 10: break

        s = x_train.shape[0] // 2
        x_train = x_train.to(device)
        x_test = x_test.to(device)

        x = torch.cat((x_test, x_train), dim=0)

        # push through model
        _, mu, x_out = model(x_test, test=True)
        _, _, adv = loss_fn(x_test, x_out, mu, label=None, test=True)

        # only look at how test data is related to train
        #iso = iso[:s,s:].mean(dim=(1))
        #adv = adv[:s]
        score = adv

        if i == 0: score_track = score
        else: score_track = torch.cat((score_track, score), dim=0)

    return ptnp(score_track)

# run ood vs id testing on all available datasets
def ood_test(model, loss_fn, args):
    dataset_names = all_datasets()
    train_loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size//2)

    score_track = {}
    for dataset in dataset_names:
        test_loader = get_loader(args.data_path, dataset, 'test', args.batch_size//2)
        score = test(model, train_loader, test_loader, loss_fn, args.device)
        score_track[dataset] = score

        print(f'{dataset}: {score.mean():.5f}')

    return score_track

if __name__ == '__main__':
    path = 'results/smaller_lmbd'
    device = 'cuda'

    # load training args
    with open(os.path.join(path, 'args.json'), 'r') as f:
        args = json.load(f)
    args = Namespace(**args)
    args.device = device # TODO: fix this
    train_dataset = args.dataset

    # load model
    model = get_autoenc(args).to(device)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

    train_loader = get_loader(args.data_path, train_dataset, 'train', args.batch_size)
    loss_fn = Loss(train_loader, args)

    # get testing performance
    score = ood_test(model, loss_fn, args)
    metric = metrics(score, args.dataset)
    print(metric)
