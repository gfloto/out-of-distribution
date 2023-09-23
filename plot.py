import os, sys
import torch
import numpy as np
from sklearn import metrics
from einops import rearrange

import seaborn as sns
import matplotlib.pyplot as plt
from utils import ptnp

# plots aucroc for in-dist vs out-of-dist
def test_plot(save_path, iso):
    fig = plt.figure(figsize=(6,6))
    fig1 = fig.add_subplot(111)

    for k, v in iso.items():
        fig1.plot(v, label=k)
    fig1.legend()
    fig1.set_title('Isometry')

    fig1.set_ylim([0,1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics.png'))
    plt.close()

# plot training loss
def loss_plot(save_path, track):
    recon_track = np.array(track['recon'])
    iso_track = np.array(track['iso'])
    center_track = np.array(track['center'])
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(12,4))
    fig1 = fig.add_subplot(131)
    fig2 = fig.add_subplot(132)
    fig3 = fig.add_subplot(133)

    fig1.plot(recon_track, 'k')
    fig2.plot(iso_track, 'r')
    fig3.plot(recon_track + iso_track, 'b')

    fig1.set_title('Reconstruction')
    fig2.set_title('Isometry')
    fig3.set_title('Total Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'recon.npy'), recon_track)
    np.save(os.path.join(save_path, 'iso.npy'), iso_track)

def diff_loss_plot(save_path, track):
    fig = plt.figure(figsize=(4,4))
    fig1 = fig.add_subplot(111)

    fig1.plot(track, 'k')
    fig1.set_title('Diffusion Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'diff_loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'diff_loss.npy'), track)

def save_sample(x, save_path, n=4):
    batch_size = x.shape[0]
    assert batch_size >= 4*n, 'batch size must be greater than 4*n'
    x = x[:4*n]

    # rearrage into grid of nxn images
    x = rearrange(x, '(b1 b2) c h w -> b1 b2 c h w', b1=4, b2=n)
    x = rearrange(x, 'b1 b2 c h w -> (b1 h) (b2 w) c')
    x = ptnp(x)

    # save image
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(111)
    ax1.imshow(x)
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# save set of images
# visualize images
def save_images(x, x_out, save_path, n=8):
    batch_size = x.shape[0]
    n = min(n, int(np.sqrt(batch_size)))
    x = x[:n*n]
    x_out = x_out[:n*n]

    # rearrage into grid of nxn images
    x = rearrange(x, '(b1 b2) c h w -> b1 b2 c h w', b1=n, b2=n)
    x = rearrange(x, 'b1 b2 c h w -> (b1 h) (b2 w) c')

    x_out = rearrange(x_out, '(b1 b2) c h w -> b1 b2 c h w', b1=n, b2=n)
    x_out = rearrange(x_out, 'b1 b2 c h w -> (b1 h) (b2 w) c')

    # convert to cpu
    x = ptnp(x)
    x_out = torch.clamp(x_out, 0, 1)
    x_out = ptnp(x_out)

    # save image
    plt.figure(figsize=(16,8))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(x)
    ax2.imshow(x_out)
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
