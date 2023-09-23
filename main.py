import os 
import torch

from args import get_args
from datasets import get_loader
from models.autoenc import get_autoenc
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
    loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size)
    model = get_autoenc(args).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = Loss(loader, args) 
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        load_path = os.path.join('results', args.resume_path)
        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))
        optim.load_state_dict(torch.load(os.path.join(load_path, 'optim.pt')))

    # main training loop
    metric_track = None
    track = {'recon': [], 'iso': [], 'center': []}
    for epoch in range(args.epochs):
        recon, iso = train(model, loader, loss_fn, optim, args)

        # update loss track
        track['recon'].append(recon)
        track['iso'].append(iso)

        # save and plot loss data
        loss_plot(save_path, track)
        with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
            for key, val in track.items():
                f.write(f'{key}: {val}\n')

        # save model, print images to view
        print('saving model and optimizer')
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
        torch.save(optim.state_dict(), os.path.join(save_path, 'optim.pt'))

        # get and save test results
        score = ood_test(model, loss_fn, args)
        metric = metrics(score, args.dataset)
        
        metric_track = update_test(metric, metric_track)
        test_plot(save_path, metric_track)

if __name__ == '__main__':
    main()
