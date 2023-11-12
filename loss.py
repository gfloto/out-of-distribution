import torch
import torch.nn as nn
from einops import rearrange

from dist_matrix import DistMatrix

class Loss(nn.Module):
    def __init__(self, loader, args):
        super(Loss, self).__init__()
        self.x_dist = DistMatrix(loader, args)
        self.recon = args.recon

        self.spheres = args.spheres
        self.lat_dim = args.lat_dim
        self.BCE = nn.BCELoss(reduction='none')

    # pairwise gram matrix
    def mu_dist(self, mu):
        assert len(mu.shape) == 2

        mu = rearrange(mu, 'b (s d) -> b s d', s=self.spheres, d=self.lat_dim)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        g = torch.einsum('aij , bij -> ab', mu, mu)

        return g
    
    # adv loss
    def adv_loss(self, mu, label, return_length=False):
        assert len(mu.shape) == 2

        mu = rearrange(mu, 'b (s d) -> b s d', s=self.spheres, d=self.lat_dim)
        norm = mu.norm(dim=-1)

        # each sphere has length (1, 2), we sum over these (so divide by spheres)
        norm = norm / norm.shape[-1]
        length = norm.sum(dim=-1) # domain" (1, 2)

        # label is 0: train, 1: gen (so subtract by one to match domains)
        length = length - 1
        if return_length:
            return length
        else:
            adv = self.BCE(length, label)
            return adv

    def forward(self, x, x_out, mu, label, test=False):
        # recon loss options
        if self.recon == 'l2':
            recon = (x - x_out).square().mean(dim=(1,2,3))
        elif self.recon == 'l1':
            recon = (x - x_out).abs().mean(dim=(1,2,3))
        else: raise ValueError('invalid reconstruction loss')

        # get tgt distance metric
        x_mtx = self.x_dist(x)
        mu_mtx = self.mu_dist(mu)

        # isometry loss
        iso = (x_mtx - mu_mtx).square()

        # adv loss
        if test: adv = self.adv_loss(mu, label, return_length=True)
        else: adv = self.adv_loss(mu, label)

        # keep batch if testing ood
        if not test: 
            recon = recon.mean()
            iso = iso.mean()
            adv = adv.mean()

        return recon, iso, adv
