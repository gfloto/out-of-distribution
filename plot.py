import os, sys
import torch
import numpy as np
from sklearn import metrics
from einops import rearrange

import seaborn as sns
import matplotlib.pyplot as plt
from utils import ptnp

# plots aucroc for in-dist vs out-of-dist
def test_plot(save_path, nll, consist):
    fig = plt.figure(figsize=(12,6))
    fig1 = fig.add_subplot(121)
    fig2 = fig.add_subplot(122)

    for k, v in nll.items():
        fig1.plot(v, label=k)
    fig1.legend()
    fig1.set_title('NLL')

    for k, v in consist.items():
        fig2.plot(v, label=k)
    fig2.legend()
    fig2.set_title('Consistency')

    fig1.set_ylim([0,1])
    fig2.set_ylim([0,1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics.png'))
    plt.close()

# plot training loss
def loss_plot(save_path, track):
    recon_track = np.array(track['recon'])
    percept_track = np.array(track['percept'])
    kld_track = np.array(track['kld'])
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(6,6))
    fig1 = fig.add_subplot(221)
    fig2 = fig.add_subplot(222)
    fig3 = fig.add_subplot(223)
    fig4 = fig.add_subplot(224)

    fig1.plot(recon_track, 'k')
    fig2.plot(kld_track, 'r')
    fig3.plot(percept_track, 'g')
    fig4.plot(recon_track + kld_track + percept_track, 'b')

    fig1.set_title('Reconstruction')
    fig2.set_title('KL-Divergence')
    fig3.set_title('Perceptual')
    fig4.set_title('Total Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'train_recon.npy'), recon_track)
    np.save(os.path.join(save_path, 'train_percept.npy'), percept_track)
    np.save(os.path.join(save_path, 'train_kld.npy'), kld_track)

def save_sample(x, save_path, n=8):
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
