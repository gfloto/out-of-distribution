import sys
import torch
import torch.nn as nn

from dist_matrix import DistMatrix, z_dist

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.x_dist = DistMatrix(args.device)
        self.recon = args.recon
        self.f = 10

    def forward(self, x, x_out, mu, test=False):
        # recon loss options
        if self.recon == 'l2':
            recon = (x - x_out).square().mean(dim=(1,2,3))
        elif self.recon == 'l1':
            recon = (x - x_out).abs().mean(dim=(1,2,3))
        else: raise ValueError('invalid reconstruction loss')

        # get tgt distance metric
        x_mtx = self.f * self.x_dist(x)
        z_mtx = z_dist(mu, mu)

        # isometry loss
        iso = (x_mtx - z_mtx).abs().mean(dim=(1))

        # center
        center = 1e-3 * mu.square().mean(dim=(1))

        # keep batch if testing ood
        if not test: 
            recon = recon.mean()
            iso = iso.mean()
            center = center.mean()

        return recon, iso, center
