import os
import torch
import numpy as np
from tqdm import tqdm

from plot import save_images
from utils import ptnp

# main training loop
def train(model, loader, loss_fn, optim, args):
    recon_track, iso_track, center_track = [], [], []

    for i, (x, _) in enumerate(tqdm(loader)):            
        x = x.to(args.device)

        # push through model
        _, mu, x_out = model(x)
        recon, iso, center = loss_fn(x, x_out, mu)
        loss = args.recon_lambda * recon + iso + center
            
        # optimize and clip gradients
        optim.zero_grad
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
                    
        recon_track.append(ptnp(recon))
        iso_track.append(ptnp(iso))
        center_track.append(ptnp(center))

    # save sample images
    if args.save_images:
        save_images(x, x_out, os.path.join('results', args.test_name, 'sample.png'))

    return np.mean(recon_track), np.mean(iso_track), np.mean(center_track) 

if __name__== '__main__':
    pass
