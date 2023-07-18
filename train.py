import sys, os
import torch
import numpy as np
from tqdm import tqdm

from dist import dist_matrix
from plot import save_images
from utils import ptnp

# main training loop
def train(model, loader, loss_fn, optim, args):
    recon_track, percept_track, kld_track = [], [], []
    for i, (x, _) in enumerate(tqdm(loader)):            
        x = x.to(args.device)

        # get distance metric
        dist = dist_matrix(x)

        # push through model
        z, mu, x_out = model(x)
        recon, percept, kld = loss_fn(x, x_out, mu)
        loss = recon + percept + kld
            
        # optimize and clip gradients
        optim.zero_grad
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
                    
        recon_track.append(ptnp(recon))
        percept_track.append(ptnp(percept))
        kld_track.append(ptnp(kld))

    # save sample images
    if args.save_images:
        save_images(x, x_out, os.path.join('results', args.test_name, 'sample.png'))

    return np.mean(recon_track), np.mean(percept_track), np.mean(kld_track)

if __name__== '__main__':
    pass
