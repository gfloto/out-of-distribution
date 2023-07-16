import sys
import torch
import torch.nn as nn
from einops import rearrange

from percept import Percept

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.tilt = args.tilt if args.tilt != None else 0
        self.dim = args.lat_dim
        self.spheres = args.spheres
        self.beta = args.beta
        self.recon = args.recon

        # perceptual loss
        self.percept_lmda = args.percept_lmda
        if self.percept_lmda != None:
            self.percept = Percept().eval()
            self.percept.to(args.device)

    def forward(self, x, x_out, mu, test=False):
        # recon loss options
        if self.recon == 'l2':
            recon = (x - x_out).square().sum(dim=(1,2,3))
        elif self.recon == 'l1':
            recon = (x - x_out).abs().sum(dim=(1,2,3))
        if self.percept_lmda != None:
            percept = self.percept(x.contiguous(), x_out.contiguous()).sum(dim=(1,2,3))
            percept = self.percept_lmda * percept
        else: percept = torch.tensor(0.0) # for plotting by default

        # kld loss options
        mu = rearrange(mu, 'b (s d) -> b s d', s=self.spheres)
        mu_norm = torch.linalg.norm(mu, dim=-1)
        kld = (mu_norm - self.tilt).pow(2).sum(dim=-1)
        kld = self.beta * kld

        # keep batch if testing ood
        if not test: 
            recon = recon.mean()
            percept = percept.mean()
            kld = kld.mean()

        return recon, percept, kld
