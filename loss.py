import torch
import torch.nn as nn

from dist_matrix import DistMatrix, z_dist

class Loss(nn.Module):
    def __init__(self, loader, args):
        super(Loss, self).__init__()
        self.x_dist = DistMatrix(loader, args)
        self.recon = args.recon
        self.metric = args.metric

    def forward(self, x, x_out, mu, test=False):
        # recon loss options
        if self.recon == 'l2':
            recon_loss_vec = (x - x_out).square()
        elif self.recon == 'l1':
            recon_loss_vec = (x - x_out).abs()
        else: raise ValueError('invalid reconstruction loss')
        recon = recon_loss_vec.mean(dim=(1,2,3))

        # get tgt distance metric
        x_mtx = self.x_dist(x)
        mu_mtx = z_dist(mu, metric=self.metric)

        # isometry loss
        iso = (x_mtx - mu_mtx).square()

        # center
        center = 1e-3 * mu.square().mean(dim=(1))

        # keep batch if testing ood
        if not test: 
            recon = recon.mean()
            iso = iso.mean()
            center = center.mean()

        return recon, iso, center
