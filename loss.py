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

    # pairwise gram matrix
    def z_dist(self, z):
        assert len(z.shape) == 2

        z = rearrange(z, 'b (s d) -> b s d', s=self.spheres, d=self.lat_dim)
        g = torch.einsum('aij , bij -> ab', z, z)

        return g

    def forward(self, x, x_out, mu, test=False):
        # recon loss options
        if self.recon == 'l2':
            recon = (x - x_out).square().mean(dim=(1,2,3))
        elif self.recon == 'l1':
            recon = (x - x_out).abs().mean(dim=(1,2,3))
        else: raise ValueError('invalid reconstruction loss')

        # get tgt distance metric
        x_mtx = self.x_dist(x)
        mu_mtx = self.z_dist(mu)

        # isometry loss
        iso = (x_mtx - mu_mtx).square()

        # keep batch if testing ood
        if not test: 
            recon = recon.mean()
            iso = iso.mean()

        return recon, iso
