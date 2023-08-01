import os 
import json
import argparse
from tqdm import tqdm

import torch 
import numpy as np
from einops import rearrange

from models.unet import Unet
from datasets import get_latents, save_latents
from diffusion_utils import Diffusion

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='l2_lat256')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')


    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--T', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--device', default='cuda', help='device being used')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'

    return args

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

def train(model, loader, optim, args):

    for i, x in enumerate(tqdm(loader)):
        x = x.to(args.device)
        t = torch.rand(size=((1,))).to(args.device)
        out = model(x, t)
    quit()

def main():
    args = get_args()
    with open(os.path.join('results', args.test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)
    save_path = os.path.join('results', args.test_name)

    # get data
    d = np.sqrt(train_args.lat_dim)
    if d.is_integer(): d = int(d)
    else: raise ValueError('latent dimension must be a perfect square')

    #save_latents(save_path, train_args.dataset, args.device) 
    z = get_latents(save_path, train_args.dataset, 'train') 
    z = rearrange(z, 'b (h w) -> b 1 h w', h=d, w=d) 
    loader = LatentLoader(z, batch_size=64, shuffle=True)

    # model
    model = Unet(dim=64, channels=1).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # diffusion helper object
    diffusion = Diffusion(args.T)

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        load_path = os.path.join('results', args.resume_path)
        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))
        optim.load_state_dict(torch.load(os.path.join(load_path, 'optim.pt')))


    # train diffusion model
    for epoch in range(args.epochs):
        train(model, loader, optim, args)
        pass

if __name__ == '__main__':
    main()