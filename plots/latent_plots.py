import os 
import torch
import numpy as np

from args import get_args
from datasets import get_loader
from models.autoenc import get_autoenc
from loss import Loss
from train import train
from plots.plot import loss_plot, test_plot
from test import ood_test
from metrics import metrics
from matplotlib import pyplot as plt

def update_test(score, score_track):
    if score_track is None:
        score_track = {}
        for k, v in score.items():
            score_track[k] = [v]
    else:
        for k, v in score.items():
            score_track[k].append(v.mean().item())

    return score_track

def plot_latent_metrics(latent_metrics, save_path):
    # plot the latent metrics as a spaghetti plot groupby latent dimension
    # latent_metrics is a dictionary of dictionaries
    # outer dictionary is keyed by latent dimension
    # inner dictionary is keyed by metric
    # each inner dictionary contains a list of metric values
    # e.g. latent_metrics[2]['recon'] is a list of reconstruction losses for latent dimension 2
    # e.g. latent_metrics[2]['iso'] is a list of isometry losses for latent dimension 2
    # e.g. latent_metrics[2]['center'] is a list of centering losses for latent dimension 2
    # also plot the 90% intervals for each metric

    # plot the latent metrics as a spaghetti plot groupby latent dimension
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(6,6))
    fig1 = fig.add_subplot(221)
    fig2 = fig.add_subplot(222)
    fig3 = fig.add_subplot(223)
    fig4 = fig.add_subplot(224)

    for lat_dim, metric_dict in latent_metrics.items():
        recon_track = np.array(metric_dict['recon'])
        iso_track = np.array(metric_dict['iso'])
        center_track = np.array(metric_dict['center'])

        # we want the frobenius norm of the iso_track, so we will take the square root of iso 
        iso_track = np.sqrt(iso_track)

        fig1.plot(recon_track, label=lat_dim)
        fig2.plot(iso_track, label=lat_dim)
        fig3.plot(center_track, label=lat_dim)
        fig4.plot(recon_track + iso_track + center_track, label=lat_dim)

    fig1.set_title('Reconstruction')
    fig2.set_title('Isometry')
    fig3.set_title('Centering')
    fig4.set_title('Total Loss')

    # plot the 90% confidence intervals for the mean of each metric
    for fig in [fig1, fig2, fig3, fig4]:
        y = fig.get_lines()[0].get_ydata()
        x = np.arange(len(y))
        fig.fill_between(x, y - 1.645 * np.std(y), y + 1.645 * np.std(y), alpha=0.2)
    
    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'latent_metrics.png'))
    plt.close()

def main():
    args = get_args()
    save_path = os.path.join('results', args.test_name)

    # dataloader, model, loss, etc
    loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size)
    latent_dims = args.lat_dims
    latent_metrics = {}
    for lat_dim in latent_dims:
        # reload objects for each latent dimension
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
        track = {'recon': [], 'iso': [], 'center': []}
        for epoch in range(args.epochs):
            recon, iso, center = train(model, loader, loss_fn, optim, args)

            # update loss track
            track['recon'].append(recon)
            track['iso'].append(iso)
            track['center'].append(center)

            # save and plot loss data
            loss_plot(save_path, track)
            with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
                for key, val in track.items():
                    f.write(f'{key}: {val}\n')

            # save model, print images to view
            print('saving model and optimizer')
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            torch.save(optim.state_dict(), os.path.join(save_path, 'optim.pt'))

        latent_metrics[lat_dim] = track

    # create line plot of the latent metrics
    # save latent metrics to .npy
    np.save(os.path.join(save_path, f'latent_metrics_dims_{latent_dims}.npy'), latent_metrics)
    plot_latent_metrics(latent_metrics, save_path)

if __name__ == '__main__':
    main()
