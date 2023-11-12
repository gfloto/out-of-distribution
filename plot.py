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
    label_track = np.array(track['label'])
    delta_track = np.array(track['delta'])
    adv_label_track = np.array(track['adv_label'])
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(9,6))
    fig1 = fig.add_subplot(231)
    fig2 = fig.add_subplot(232)
    fig3 = fig.add_subplot(233)
    fig4 = fig.add_subplot(234)
    fig5 = fig.add_subplot(235)
    fig6 = fig.add_subplot(236)

    fig1.plot(iso_track, 'k')
    fig2.plot(label_track, 'b')
    fig3.plot(recon_track, 'g')
    fig4.plot(delta_track, 'k')
    fig5.plot(adv_label_track, 'b')
    fig6.plot(recon_track + iso_track, 'r')

    fig1.set_title('Isometry')
    fig2.set_title('Label')
    fig3.set_title('Reconstruction')
    fig4.set_title('Delta')
    fig5.set_title('Adv Label')
    fig6.set_title('Total Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'recon.npy'), recon_track)
    np.save(os.path.join(save_path, 'iso.npy'), iso_track)
    np.save(os.path.join(save_path, 'adv.npy'), adv_label_track)

# plot for info diff training loss
def infodiff_loss_plot(save_path, enc_track, dec_track):
    fig = plt.figure(figsize=(4,4))
    fig1 = fig.add_subplot(111)

    fig1.plot(enc_track, 'k', label='Encoder')
    fig1.plot(dec_track, 'r', label='Decoder')
    fig1.legend()
    fig1.set_title('Info Diffusion Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'infodiff_loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'enc_loss.npy'), enc_track)
    np.save(os.path.join(save_path, 'dec_loss.npy'), dec_track)

def diff_loss_plot(save_path, track):
    fig = plt.figure(figsize=(4,4))
    fig1 = fig.add_subplot(111)

    fig1.plot(track, 'k')
    fig1.set_title('Diffusion Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'diff_loss.png'))
    plt.close()

    np.save(os.path.join(save_path, 'diff_loss.npy'), track)

def save_sample(x, save_path, n=12):
    batch_size = x.shape[0]
    assert batch_size >= 4*n, 'batch size must be greater than 4*n'
    x = x[:4*n]

    # rearrage into grid of nxn images
    x = rearrange(x, '(b1 b2) c h w -> b1 b2 c h w', b1=4, b2=n)
    x = rearrange(x, 'b1 b2 c h w -> (b1 h) (b2 w) c')
    x = torch.clamp(x, 0, 1)
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
    x = torch.clamp(x, 0, 1)
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

# save batch of images, with titles
def save_gen(x, delta, hard, save_path):
    x = torch.clamp(x, 0, 1)
    x = ptnp(x)
    x = rearrange(x, 'b c h w -> b h w c')

    # save 4x4 grid
    fig = plt.figure(figsize=(16,16))
    for i in range(16):
        ax = fig.add_subplot(4,4,i+1)
        ax.imshow(x[i])
        ax.set_title(f'delta: {delta[i]:.2f}, hard: {hard[i]:.2f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

