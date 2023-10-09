import os 
import json
import argparse
from tqdm import tqdm

import torch 
import numpy as np
from einops import rearrange

from models.basic import Basic
from models.transformer import Transformer

from utils import ptnp
from models.autoenc import get_autoenc
from datasets import get_latents, save_latents
from diffusion_utils import Diffusion
from plot import save_sample, diff_loss_plot
from datasets import all_datasets
from metrics import metrics

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='lat_36')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')

    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--T', type=int, default=500, help='number of diffusion steps')
    parser.add_argument('--device', default='cuda', help='device being used')

    parser.add_argument('--gen_lats', action='store_true', help='generate latents for training') 
    parser.add_argument('--sample_freq', type=int, default=10, help='frequency of saving model')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'

    return args

# convenience function for getting latent loader
def get_lat_loader(save_path, dataset, mode, batch_size=64):
    assert mode in ['train', 'test']
    if mode == 'test': batch_size = 32

    z = get_latents(save_path, dataset, mode) 
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

from plot import save_images
def train(model, autoenc, loader, diffusion, optim, args):

    hit = False
    loss_track = []
    model.train()
    for i, x0 in enumerate(tqdm(loader)):

        x0 = x0.to(args.device)
        t_ = np.random.randint(1, args.T)
        t = torch.tensor([t_]).float().to(args.device) / args.T

        # get xt
        xt, eps = diffusion.q_xt_x0(x0, t_)

        # push through model
        pred = model(xt, t)
        loss = (pred - eps).square().mean()

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()        

        loss_track.append(ptnp(loss))

        # decode predictions to compare to original
        if t.item() < 0.1 and not hit:
            hit = True
            pred_x0 = diffusion.p_x0_xt(xt, pred, t_)
            pred_out = autoenc.decode(pred_x0) 
            x0_out = autoenc.decode(x0)
            save_images(x0_out, pred_out, os.path.join('results', args.test_name, 'diffusion/decode.png'))
    
    return np.mean(loss_track)

def test(model, diffusion, train_loader, test_loader, device):
    model.eval()

    for i, x0 in enumerate(tqdm(test_loader)):
        if i > 4: break

        x0 = x0.to(device)
        score = diffusion.likelihood(x0, model)

        if i == 0:
            score_track = score
        else:
            score_track = torch.cat((score_track, score), dim=0)

    return ptnp(score_track) 

# run ood vs id testing on all available datasets
def ood_test(model, diffusion, train_args, args):
    dataset_names = all_datasets()
    save_path = os.path.join('results', args.test_name, 'diffusion')
    train_loader = get_lat_loader(save_path, train_args.dataset, 'train', args.batch_size)


    score_track = {}
    for dataset in dataset_names:
        test_loader = get_lat_loader(save_path, dataset, 'test', args.batch_size)
        score = test(model, diffusion, train_loader, test_loader, args.device)
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
    loader = get_lat_loader(save_path, train_args.dataset, 'train', args.batch_size)

    # model
    model = Basic(dim=36).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # diffusion helper object
    diffusion = Diffusion(args.T, args.device)

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        model.load_state_dict(torch.load(os.path.join(args.resume_path, 'diff_model.pt')))
        optim.load_state_dict(torch.load(os.path.join(args.resume_path, 'diff_optim.pt')))

    # load autoencoder model
    autoenc = get_autoenc(train_args).to(args.device) 
    autoenc.load_state_dict(
        torch.load(os.path.join('results', train_args.test_name, 'model.pt'))
    )

    # train diffusion model
    loss_track = []
    for epoch in range(args.epochs):
        loss = train(model, autoenc, loader, diffusion, optim, args)
        loss_track.append(loss)

        ## save loss plot
        diff_loss_plot(save_path, loss_track)

        # sample batch
        if epoch % args.sample_freq == 0 and epoch > 0:
            print('sampling')
            sample = diffusion.sample(model, autoenc, dim=loader.dim)
            save_sample(sample, os.path.join(save_path, f'diff_sample.png'))

            # ood test
            print('ood test')
            score = ood_test(model, diffusion, train_args, args)
            metric = metrics(score, train_args.dataset)
            print(metric)

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, 'diff_model.pt'))
        torch.save(optim.state_dict(), os.path.join(save_path, 'diff_optim.pt'))

if __name__ == '__main__':
    main()