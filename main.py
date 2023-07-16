import sys, os 
import torch
import numpy as np 

from args import get_args
from datasets import get_loader
from models import get_model
from loss import Loss
from train import train
from plot import loss_plot, test_plot
from test import ood_test
from metrics import metrics

def update_test(score, score_track):
    if score_track is None:
        score_track = {}
        for k, v in score.items():
            score_track[k] = [v]
    else:
        for k, v in score.items():
            score_track[k].append(v.mean().item())

    return score_track

def main():
    args = get_args()
    save_path = os.path.join('results', args.test_name)

    # dataloader, model, loss, etc
    loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size, args.workers)
    model = get_model(args.lat_dim).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = Loss(args) 
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'latent dim: {args.lat_dim}, spheres: {args.spheres}, tilt: {args.tilt}')

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pt')))
        optim.load_state_dict(torch.load(os.path.join(save_path, 'optim.pt')))

    # main training loop
    nll_track = None; consist_track = None
    track = {'recon': [], 'percept': [], 'kld': []}
    for epoch in range(args.epochs):
        recon, percept, kld = train(model, loader, loss_fn, optim, args)

        # update loss track
        track['recon'].append(recon)
        track['percept'].append(percept)
        track['kld'].append(kld)

        # save and plot loss data
        loss_plot(save_path, track)
        with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
            for key, val in track.items():
                f.write(f'{key}: {val}\n')

        # save model, print images to view
        if epoch == args.epochs-1 or epoch % args.save_freq == 0:
            print('saving model and optimizer')
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            torch.save(optim.state_dict(), os.path.join(save_path, 'optim.pt'))

            # get and save test results
            nll = ood_test(model, loss_fn, args)
            nll_metric = metrics(nll, args.dataset)
            nll_track = update_test(nll_metric, nll_track)

            consist = ood_test(model, loss_fn, args, consistent_score=True)
            consist_metric = metrics(consist, args.dataset)
            consist_track = update_test(consist_metric, consist_track)
            test_plot(save_path, nll_track, consist_track)

if __name__ == '__main__':
    main()