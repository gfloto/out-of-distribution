import json
import os
from scipy import stats 
import torch
import numpy as np
import statsmodels.api as sm

from args import get_args
from datasets import get_loader
from models.autoenc import get_autoenc
from loss import Loss
from train import train
from plot import loss_plot, test_plot
from test import ood_test
from metrics import metrics
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from pygam import LinearGAM
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from utils import ptnp

def update_test(score, score_track):
    if score_track is None:
        score_track = {}
        for k, v in score.items():
            score_track[k] = [v]
    else:
        for k, v in score.items():
            score_track[k].append(v.mean().item())

    return score_track


def lowess_with_bootstrap_ci(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw={}):
    # Perform initial lowess smoothing
    smoothed = sm.nonparametric.lowess(y, x, **lowess_kw)[:, 1]
    
    # Perform bootstrap resampling and re-fitting
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample_indices = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample_indices]
        sampled_y = y[sample_indices]
        smoothed_values[i] = sm.nonparametric.lowess(sampled_y, sampled_x, **lowess_kw)[:, 1]
    
    # Compute the confidence intervals
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]
    
    return smoothed, bottom, top


def compute_model_losses(loader, model, args, loss_fn):
    # pytorch arrays 
    recon_losses, iso_losses, center_losses = [], [], []

    for i, (x, _) in enumerate(tqdm(loader)):            
        x = x.to(args.device)

        # push through model
        _, mu, x_out = model(x)
        recon, iso, center = loss_fn(x, x_out, mu, test=True)
        
        recon_losses.append(ptnp(recon).ravel())
        iso_losses.append(ptnp(iso).ravel())
        center_losses.append(ptnp(center).ravel())

    # concatenate into single array
    recon_losses = np.concatenate(recon_losses)
    iso_losses = np.sqrt(np.concatenate(iso_losses))
    center_losses = np.concatenate(center_losses)
    return recon_losses, iso_losses, center_losses

def compute_confidence_interval(recon_losses, iso_losses, center_losses, confidence_levels):

    # compute confidence intervals
    recon_CIs, iso_CIs, center_CIs = {}, {}, {}
    for confidence in confidence_levels:
        recon_CIs[confidence] = confidence_interval_helper(recon_losses, confidence)
        iso_CIs[confidence] = confidence_interval_helper(iso_losses, confidence)
        center_CIs[confidence] = confidence_interval_helper(center_losses, confidence)

    return recon_CIs, iso_CIs, center_CIs

def confidence_interval_helper(data, confidence):
    # compute the confidence interval for data
    # data is a numpy array

    # compute mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    confidence_interval = stats.norm.interval(confidence, loc=mean, scale=std)
    return mean, confidence_interval

def build_plot(intervals, recon_losses, iso_losses, center_losses, method, sorted_lat_dims, save_path, confidence_levels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for key, color in zip(["Recon", "Iso", "Center"], ['black', 'red', 'green']):
        label_created = False
        for confidence_level in confidence_levels:
            x = []
            x_dups = []
            y_mean = []
            lower_ci = []
            upper_ci = []
            loss_values = []

            for lat_dim in sorted_lat_dims:
                recon_mean, recon_ci, iso_mean, iso_ci, center_mean, center_ci = intervals[(lat_dim, confidence_level)]

                x.append(lat_dim)
                if key == "Recon":
                    y_mean.append(recon_mean)
                    lower_ci.append(recon_ci[0])
                    upper_ci.append(recon_ci[1])
                    loss_values.append(recon_losses[lat_dim])
                    # x_dups is the lat_dim duplicated for each loss value
                    x_dups.extend([lat_dim] * len(recon_losses[lat_dim]))
                elif key == "Iso":
                    y_mean.append(iso_mean)
                    lower_ci.append(iso_ci[0])
                    upper_ci.append(iso_ci[1])
                    loss_values.append(iso_losses[lat_dim])
                    # x_dups is the lat_dim duplicated for each loss value
                    x_dups.extend([lat_dim] * len(iso_losses[lat_dim]))
                elif key == "Center":
                    y_mean.append(center_mean)
                    lower_ci.append(center_ci[0])
                    upper_ci.append(center_ci[1])
                    loss_values.append(center_losses[lat_dim])
                    # x_dups is the lat_dim duplicated for each loss value
                    x_dups.extend([lat_dim] * len(center_losses[lat_dim]))

            x = np.array(x)
            x_dups = np.array(x_dups).flatten()
            loss_values = np.array(loss_values).flatten()
            y_mean = np.array(y_mean)
            lower_ci = np.array(lower_ci)
            upper_ci = np.array(upper_ci)

            if not label_created:
                label = f"{key}"

            label_created = True

            if method == "spline":
                xnew = np.linspace(min(x), max(x), 300)
                y_spl = make_interp_spline(x, y_mean, k=3)
                upper_ci_spl = make_interp_spline(x, upper_ci, k=3)
                lower_ci_spl = make_interp_spline(x, lower_ci, k=3)
                ynew = y_spl(xnew)
                upper_ci = upper_ci_spl(xnew)
                lower_ci = lower_ci_spl(xnew)
                ax.plot(xnew, ynew, color=color, label=f"{label} Mean")
                # add scatter plot of points
                # ax.scatter(x_dups, loss_values, color=color, s=1)
                ax.fill_between(xnew, lower_ci, upper_ci, color=color, alpha=0.2)

            elif method == "lowess":
                xvals = np.linspace(min(x), max(x), len(x_dups))
                lowess_kw = {
                    "xvals": xvals
                }
                smoothed, bottom, top = lowess_with_bootstrap_ci(x_dups, loss_values, x_dups, conf_interval=1-confidence_level,lowess_kw=lowess_kw)
                ax.plot(xvals, smoothed, color=color, label=f"{label} Mean")
                ax.fill_between(xvals, bottom, top, color=color, alpha=0.2)

            elif method == "gam":
                gam = LinearGAM().fit(x_dups, loss_values)
                XX = gam.generate_X_grid(term=0)
                preds = gam.predict(XX)
                conf_int = gam.confidence_intervals(XX, width=confidence_level)
                ax.plot(XX, preds, color=color, label=f"{label} Mean")
                ax.fill_between(XX[:, 0], conf_int[:, 0], conf_int[:, 1], color=color, alpha=0.2)

    ax.set_title('Latent Metrics')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'latent_metrics_{method}.png'))

def plot_latent_metrics(latent_models, loader, args, loss, save_path):
    confidence_levels = args.ci_confidence_levels
    intervals = {}

    # Sorting latent dimensions if they are not sorted
    latent_dims = list(latent_models.keys())
    recon_losses, iso_losses, center_losses = {}, {}, {}

    # Check if we can load the losses from a previous run
    if os.path.exists(os.path.join(save_path, 'recon_losses.npy')):
        recon_losses = np.load(os.path.join(save_path, 'recon_losses.npy'), allow_pickle=True).item()
    
    if os.path.exists(os.path.join(save_path, 'iso_losses.npy')):
        iso_losses = np.load(os.path.join(save_path, 'iso_losses.npy'), allow_pickle=True).item()
    
    if os.path.exists(os.path.join(save_path, 'center_losses.npy')):
        center_losses = np.load(os.path.join(save_path, 'center_losses.npy'), allow_pickle=True).item()

    # Compute the losses if they are not already computed
    if not recon_losses or not iso_losses or not center_losses:
        for lat_dim in latent_dims:
            model = latent_models[lat_dim]
            recon_losses[lat_dim], iso_losses[lat_dim], center_losses[lat_dim] = compute_model_losses(loader, model, args, loss)

        # Save the losses as numpy arrays
        np.save(os.path.join(save_path, 'recon_losses.npy'), recon_losses)
        np.save(os.path.join(save_path, 'iso_losses.npy'), iso_losses)
        np.save(os.path.join(save_path, 'center_losses.npy'), center_losses)

    # Collect metrics and confidence intervals first
    for lat_dim in latent_dims:
        dim_recon_losses, dim_iso_losses, dim_center_losses = recon_losses[lat_dim], iso_losses[lat_dim], center_losses[lat_dim]
        recon_CIs, iso_CIs, center_CIs = compute_confidence_interval(dim_recon_losses, dim_iso_losses, dim_center_losses, confidence_levels)
        for confidence_level in confidence_levels:
            recon_mean, recon_ci = recon_CIs[confidence_level]
            iso_mean, iso_ci = iso_CIs[confidence_level]
            center_mean, center_ci = center_CIs[confidence_level]

            intervals[(lat_dim, confidence_level)] = (recon_mean, recon_ci, iso_mean, iso_ci, center_mean, center_ci)

    # Build the plot
    build_plot(intervals, recon_losses, iso_losses, center_losses, args.ci_iterpolate_method, latent_dims, save_path, confidence_levels)


def main():
    args = get_args()
    save_path = os.path.join('results', args.test_name)

    # dataloader, model, loss, etc
    loader = get_loader(args.data_path, args.dataset, 'train', args.batch_size)
    loss_fn = Loss(loader, args)

    latent_dims = args.lat_dims
    latent_models = {}
    for lat_dim in latent_dims:
        # overwrite lat_dim in args
        args.lat_dim = lat_dim
        print(f'Latent dimension: {lat_dim}')
        # reload objects for each latent dimension
        model = get_autoenc(args).to(args.device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

        # load model and optimizer if resuming training
        if args.resume_path is not None:
            print('Trying to load model and optimizer from checkpoint')
            load_path = os.path.join('results', args.resume_path)
            if os.path.exists(os.path.join(load_path, f'model_ld_{lat_dim}.pt')):
                model.load_state_dict(torch.load(os.path.join(load_path, f'model_ld_{lat_dim}.pt')))
                latent_models[lat_dim] = model
            if os.path.exists(os.path.join(load_path, f'optim_ld_{lat_dim}.pt')):
                optim.load_state_dict(torch.load(os.path.join(load_path, f'optim_ld_{lat_dim}.pt')))
                continue    

        # main training loop
        track = {
                    'recon': [], 
                    'iso': [], 
                    'center': [], 
                 }
        for epoch in range(args.epochs):
            recon, iso, center = train(model, loader, loss_fn, optim, args)

            # update loss track
            track['recon'].append(recon)
            track['iso'].append(iso)
            track['center'].append(center)

            # save and plot loss data
            lat_dim_path = os.path.join(save_path, str(lat_dim))
            # mkdir if doesn't exist
            if not os.path.exists(lat_dim_path):
                os.makedirs(lat_dim_path)

            loss_plot(lat_dim_path, track)
            with open(os.path.join(lat_dim_path, 'loss.txt'), 'w') as f:
                for key, val in track.items():
                    f.write(f'{key}: {val}\n')

            # save model, print images to view
            print('saving model and optimizer')
            torch.save(model.state_dict(), os.path.join(save_path, f'model_ld_{lat_dim}.pt'))
            torch.save(optim.state_dict(), os.path.join(save_path, f'optim_ld_{lat_dim}.pt'))

        latent_models[lat_dim] = model

    # create line plot of the latent metrics
    plot_latent_metrics(latent_models, loader, args, loss_fn, save_path)

if __name__ == '__main__':
    main()
