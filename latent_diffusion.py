import os 
import json
import argparse
from tqdm import tqdm

import torch 
import numpy as np
from einops import rearrange

from utils import ptnp
from models.autoenc import get_autoenc
from models.transformer import Transformer
from datasets import get_latents, save_latents
from diffusion_utils import Diffusion
from plot import save_sample, diff_loss_plot
from datasets import all_datasets

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='mnist_36')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')

    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--T', type=int, default=200, help='number of diffusion steps')
    parser.add_argument('--device', default='cuda', help='device being used')

    parser.add_argument('--gen_lats', action='store_true', help='generate latents for training') 
    parser.add_argument('--sample_freq', type=int, default=10, help='frequency of saving model')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'

    return args

# convenience function for getting latent loader
def get_lat_loader(save_path, dataset, batch_size=64):
    z = get_latents(save_path, dataset, 'train') 

    # reshape to be square for transformer / unet
    d = np.sqrt(z.shape[1])
    if d.is_integer(): d = int(d)
    else: raise ValueError('latent dimension must be a perfect square')

    z = rearrange(z, 'b (h w) -> b h w', h=d, w=d) 
    loader = LatentLoader(z, batch_size=batch_size, shuffle=True)

    return loader

class LatentLoader:
    def __init__(self, x, batch_size, shuffle=True):
        self.x = x
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_batch = int(np.ceil(self.x.shape[0] / self.batch_size))

        self.i = 0
        if self.shuffle:
            self.ind = np.random.permutation(np.arange(x.shape[0]))
        else:
            self.ind = np.arange(x.shape[0])

        self.dim = x.shape[1]
        assert x.shape[1] == x.shape[2]
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n_batch - 1:
            ind = self.ind[self.i * self.batch_size : (self.i + 1) * self.batch_size]
        elif self.i == self.n_batch - 1:
            ind = self.ind[self.i * self.batch_size : ]
        else:
            self.i = 0
            if self.shuffle: self.shuffle_ind()
            raise StopIteration

        x_ = self.x[ind]
        self.i += 1

        return x_ 

    def shuffle_ind(self):
        self.ind = np.random.permutation(self.ind)

def train(model, loader, diffusion, optim, args):

    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):

        x0 = x0.to(args.device)
        t_ = np.random.randint(0, args.T)
        t = torch.tensor([t_]).float().to(args.device) / args.T

        # get xt
        xt, eps = diffusion.q_xt_x0(x0, t_)

        # push through model
        pred = model(xt, t)
        loss = (pred - eps).square().mean()
        #loss = (xt - x0+pred).square().mean()

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()        

        loss_track.append(ptnp(loss))
    
    return np.mean(loss_track)

def test(model, loader, diffusion, args):
    model.eval()

    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        if i > 5: break

    return np.mean(loss_track)

# run ood vs id testing on all available datasets
def ood_test(model, save_path, train_dataset, d):
    z = get_latents(save_path, train_dataset, 'train') 
    z = rearrange(z, 'b (h w) -> b h w', h=d, w=d) 
    train_loader = LatentLoader(z, batch_size=64, shuffle=True)

    dataset_names = all_datasets()

    score_track = {}
    for dataset in dataset_names:
        test_loader = get_loader(args.data_path, dataset, 'test', args.batch_size//2)
        score = test(model, train_loader, test_loader, loss_fn, args.device)
        score_track[dataset] = score

        print(f'{dataset}: {score.mean():.5f}')

    return score_track

def main():
    args = get_args()
    with open(os.path.join('results', args.test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)
    save_path = os.path.join('results', args.test_name, 'diffusion')
    os.makedirs(save_path, exist_ok=True)

    # push datasets through model to get latents
    if args.gen_lats:
        save_latents(train_args, args.device) 
    
    # get latent loader
    loader = get_lat_loader(save_path, train_args.dataset, batch_size=64)

    # model
    model = Transformer(dim=loader.dim).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # diffusion helper object
    diffusion = Diffusion(args.T, args.device)

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        model.load_state_dict(torch.load(os.path.join(save_path, 'diff_model.pt')))
        optim.load_state_dict(torch.load(os.path.join(save_path, 'diff_optim.pt')))

    # load autoencoder model
    autoenc = get_autoenc(train_args).to(args.device) 
    autoenc.load_state_dict(
        torch.load(os.path.join('results', train_args.test_name, 'model.pt'))
    )

    # train diffusion model
    loss_track = []
    for epoch in range(args.epochs):
        loss = train(model, loader, diffusion, optim, args)
        loss_track.append(loss)

        # save loss plot
        diff_loss_plot(save_path, loss_track)


        # sample batch
        if epoch % args.sample_freq == 0 and epoch > 0:
            print('sampling')
            sample = diffusion.sample(model, autoenc, dim=loader.dim)
            save_sample(sample, os.path.join(save_path, f'diff_sample_{epoch}.png'))

        # ood test
        #print('ood test')
        #score = ood_test(model, save_path, train_args.dataset, loader.d)

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, 'diff_model.pt'))

if __name__ == '__main__':
    main()