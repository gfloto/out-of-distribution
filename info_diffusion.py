import os 
import json
import argparse
from tqdm import tqdm

import torch 
import numpy as np

from models.basic import Basic

from utils import ptnp
from models.autoenc import get_autoenc
from datasets import  get_lat_loader
from diffusion_utils import Diffusion
from plot import save_sample, diff_loss_plot
from datasets import all_datasets
from metrics import metrics

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='lat_36')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--T', type=int, default=10, help='number of diffusion steps')
    parser.add_argument('--device', default='cuda', help='device being used')

    parser.add_argument('--sample_freq', type=int, default=10, help='frequency of saving model')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'

    return args

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
        if torch.rand(1) > 0.5:
            # normal diffusion
            lmbda = 1
            xt, eps = diffusion.q_xt_x0(x0, t_)
        else:
            # negative sampling
            lmbda = 0.5
            x0 = torch.randn_like(x0)
            x0 /= x0.norm(dim=-1, keepdim=True)
            xt, eps = diffusion.q_xt_x0(x0, t_)

            # length goes from 1->2
            a = diffusion.alpha_bars[t_].sqrt()
            x0 *= 2 - a

        # push through model
        pred = model(xt, t)
        loss = lmbda * (pred - x0).square().mean()

        # backprop with grad clipping
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()   

        loss_track.append(ptnp(loss))

        # decode predictions to compare to original
        if t.item() < 0.05 and not hit:
            hit = True
            #pred_x0 = diffusion.p_x0_xt(xt, pred, t_)
            pred_x0 = pred
            pred_out = autoenc.decode(pred_x0) 
            x0_out = autoenc.decode(x0)
            save_images(x0_out, pred_out, os.path.join('results', args.test_name, 'diffusion/decode.png'))
    
    return np.mean(loss_track)

def test(model, diffusion, train_loader, test_loader, device):
    model.eval()

    for i, x0 in enumerate(tqdm(test_loader)):
        if i > 5: break

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
        test_loader, z_loss = get_lat_loader(save_path, dataset, 'test', args.batch_size)
        score = test(model, diffusion, train_loader, test_loader, args.device)
        score_track[dataset] = score
        #score_track[dataset] = torch.tensor(score) + z_loss[:score.shape[0]]

        print(f'{dataset}: {score.mean():.5f}')

    return score_track

def main():
    args = get_args()
    with open(os.path.join('results', args.test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)
    save_path = os.path.join('results', args.test_name, 'info_diffusion')
    os.makedirs(save_path, exist_ok=True)

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