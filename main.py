import os 
import torch

from args import get_args, save_args
from datasets import get_loader
from loss import Loss
from train import train
from plot import loss_plot, test_plot
from test import ood_test
from metrics import metrics
from models.autoenc import get_autoenc
from models.unet import Unet

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

    # get loader which is used to find mean and std of lpips to normalize train data
    loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size)
    loss_fn = Loss(loader, args) 

    args.lpips_mean = loss_fn.x_dist.mean.item()
    args.lpips_std = loss_fn.x_dist.std.item()
    save_args(args)

    # dataloader, model, loss, etc
    adv_model = Unet(dim=128).to(args.device)
    model = get_autoenc(args).to(args.device)
    optim_model = torch.optim.Adam(model.parameters(), lr=0.5*args.lr)
    optim_adv = torch.optim.Adam(adv_model.parameters(), lr=args.lr)
    print(f'number of model parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'number of adv model parameters: {sum(p.numel() for p in adv_model.parameters())}')

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        load_path = os.path.join('results', args.resume_path)
        adv_model.load_state_dict(torch.load(os.path.join(load_path, 'adv_model.pt')))
        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))
        optim_model.load_state_dict(torch.load(os.path.join(load_path, 'optim_model.pt')))
        optim_adv.load_state_dict(torch.load(os.path.join(load_path, 'optim_adv.pt')))

    # main training loop
    metric_track = None
    track = {'recon': [], 'iso': [], 'label': [], 'delta': [], 'adv_label': []}
    for epoch in range(args.epochs):
        recon, iso, label, delta, adv_label = train(model, adv_model, loader, loss_fn, optim_model, optim_adv, args)

        # update loss track
        track['recon'].append(recon)
        track['iso'].append(iso)
        track['label'].append(label)
        track['delta'].append(delta)
        track['adv_label'].append(adv_label)

        # save and plot loss data
        loss_plot(save_path, track)
        with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
            for key, val in track.items():
                f.write(f'{key}: {val}\n')

        # save model, print images to view
        if epoch % 10 == 0 and epoch > 0:
            print('saving model and optimizer')
            torch.save(adv_model.state_dict(), os.path.join(save_path, 'adv_model.pt'))
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            torch.save(optim_model.state_dict(), os.path.join(save_path, 'optim_model.pt'))
            torch.save(optim_adv.state_dict(), os.path.join(save_path, 'optim_adv.pt'))

            # get and save test results
            #score = ood_test(model, loss_fn, args)
            #metric = metrics(score, args.dataset)
            
            #metric_track = update_test(metric, metric_track)
            #test_plot(save_path, metric_track)

if __name__ == '__main__':
    main()
