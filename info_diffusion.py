import os 
import json
import argparse
from tqdm import tqdm
import torch 
import math
import numpy as np

from utils import ptnp
from models.basic import Basic
from models.autoenc import get_autoenc
from datasets import  get_lat_loader
from plot import infodiff_loss_plot
from datasets import all_datasets
from metrics import metrics
from plot import save_images
from diffusion_utils import ComposedDiffusion

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_name', default='lat_36')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')
    parser.add_argument('--resume_step', type=int, default=None, help='step to resume training from')

    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=4e-6, help='learning rate')
    parser.add_argument('--T', type=int, default=50, help='number of diffusion steps')
    parser.add_argument('--device', default='cuda', help='device being used')

    parser.add_argument('--sample_freq', type=int, default=5, help='frequency of saving model')

    args = parser.parse_args()

    # asserts
    assert args.test_name is not None, 'enter a test name'
    assert not (args.resume_step is None and args.resume_path is not None), 'enter resume step and resume path!'
    if args.resume_step is not None:
        assert args.resume_step >= 0 and args.resume_step < args.T, 'invalid resume step'

    return args

def test(comp_diff, train_loader, test_loader, device):
    for i, x0 in enumerate(test_loader):
    #for i, x0 in enumerate(tqdm(test_loader)):
        #if i > 5: break

        x0 = x0.to(device)
        score = comp_diff.ood(x0)

        if i == 0:
            score_track = score
        else:
            score_track = torch.cat((score_track, score), dim=0)

    return ptnp(score_track) 

# run ood vs id testing on all available datasets
def ood_test(comp_diff, train_args, args):
    dataset_names = all_datasets()
    save_path = os.path.join('results', args.test_name)
    train_loader = get_lat_loader(save_path, train_args.dataset, 'train', args.batch_size)

    score_track = {}
    for dataset in dataset_names:
        test_loader, z_loss = get_lat_loader(save_path, dataset, 'test', args.batch_size)
        score = test(comp_diff, train_loader, test_loader, args.device)
        score_track[dataset] = score
        #score_track[dataset] = torch.tensor(score) + z_loss[:score.shape[0]]

        #print(f'{dataset}: {score.mean():.5f}')

    return score_track

def train(encoder, decoder, autoenc, comp_diff, loader, optim, t_ind, args):
    hit = False
    enc_track, dec_track = [], []
    encoder.train()
    decoder.train()

    t = torch.tensor([(t_ind+1) / args.T]).float().to(args.device)
    tm1 = torch.tensor([t_ind / args.T]).float().to(args.device)
    for i, x0 in enumerate(tqdm(loader)):
        # option to negative sample
        if torch.rand(1) > 0.5 or i == len(loader) - 1: # normal diffusion
            # forward pass composition
            x0 = x0.to(args.device)

            lmbda = 1
            #mu = (1-t)*x0 + t*3
            mu_norm = 1-t

            xtm1 = comp_diff.forward(
                x0 + 0.01 * torch.randn_like(x0)
            )

        else: # negative sampling
            x0 = torch.randn_like(x0).to(args.device)
            x0 /= x0.norm(dim=1, keepdim=True)

            lmbda = 2
            #mu = (1-t)*x0 - t*3
            mu_norm = 1+5*t

            xtm1 = comp_diff.forward(
                x0 + 0.01 * torch.randn_like(x0)
            )

        # encode (define mu relative to ideal value)
        if t_ind < args.T - 1:
            pred_mu = encoder(xtm1, t)
        else:
            pred_mu = torch.zeros_like(xtm1).to(args.device)
        z = pred_mu + t * torch.randn_like(pred_mu)

        # decode
        pred_x0 = decoder(z, t)

        # loss
        beta = 1 / (-math.e * t * t.log())
        enc_loss = beta * (mu_norm*x0 - pred_mu).square().mean()
        #enc_loss = beta * (mu - pred_mu).square().mean()

        dec_loss = (x0 - pred_x0).square().mean()
        loss = lmbda * (enc_loss + dec_loss)

        # backprop with grad clipping
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
        optim.step()   

        enc_track.append(ptnp(enc_loss))
        dec_track.append(ptnp(dec_loss))

    # sample batch
    xtm1 = comp_diff.forward(x0)
    xt = encoder(xtm1, t) + t * torch.randn_like(xtm1)

    f = tm1 if t_ind > 0 else 0.01
    xtm1 = decoder(xt, t) + f * torch.randn_like(xt)
    sample = comp_diff.sample(xtm1)

    pred_out = autoenc.decode(sample) 
    x0_out = autoenc.decode(x0)
    save_path = os.path.join('results', args.test_name, 'info_diffusion', f't_{t_ind}', 'sample.png')
    save_images(x0_out, pred_out, save_path)
    
    return np.mean(enc_track), np.mean(dec_track)

def main():
    args = get_args()
    with open(os.path.join('results', args.test_name, 'args.json'), 'r') as f:
        train_args = json.load(f) 
        train_args = argparse.Namespace(**train_args)
    save_path = os.path.join('results', args.test_name, 'info_diffusion')
    os.makedirs(save_path, exist_ok=True)

    # get latent loader
    load_path = os.path.join('results', train_args.test_name)
    loader = get_lat_loader(load_path, train_args.dataset, 'train', args.batch_size)

    # train diffusion model
    comp_diff = ComposedDiffusion(args.T)
    t0 = args.resume_step if args.resume_step is not None else 0
    for t in range(t0, args.T):
        enc_track, dec_track = [], []
        print(f'\nt: {t}')

        # model
        encoder = Basic(dim=36).to(args.device)
        decoder = Basic(dim=36).to(args.device)
        optim = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()
        ), lr=args.lr)

        # load autoencoder model
        autoenc = get_autoenc(train_args).to(args.device) 
        autoenc.load_state_dict(
            torch.load(os.path.join('results', train_args.test_name, 'model.pt'))
        )

        # load model and optimizer if resuming training
        if args.resume_step is not None:
            print('loading model and optimizer from checkpoint')
            load_path = os.path.join('results', args.resume_path, 'info_diffusion', f't_{t}')
            encoder.load_state_dict(torch.load(os.path.join(load_path, 'encoder.pt')))
            decoder.load_state_dict(torch.load(os.path.join(load_path, 'decoder.pt')))
            optim.load_state_dict(torch.load(os.path.join(load_path, 'optim.pt')))

        save_path = os.path.join('results', args.test_name, 'info_diffusion', f't_{t}')
        os.makedirs(save_path, exist_ok=True)

        # train q(x_t | x_{t+1}, t)
        for epoch in range(args.epochs):
            enc_loss, dec_loss = train(
                encoder, decoder, autoenc, 
                comp_diff, loader, optim, t, args
            )

            print(f'epoch {epoch} | enc_loss: {enc_loss:.5f} | dec_loss: {dec_loss:.5f}')
            enc_track.append(enc_loss)
            dec_track.append(dec_loss)

            # save loss plot
            infodiff_loss_plot(save_path, enc_track, dec_track) 

            # save models
            if (epoch % args.sample_freq == 0 and epoch > 0) or epoch == args.epochs - 1:
                print('saving model')
                torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pt'))
                torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pt'))
                torch.save(optim.state_dict(), os.path.join(save_path, 'optim.pt'))

                # perform ood test
                print('ood test')
                comp_diff.add(encoder, decoder)

                score = ood_test(comp_diff, train_args, args)
                metric = metrics(score, train_args.dataset)
                print(metric)

                comp_diff.remove()

        # add to composition 
        comp_diff.add(encoder, decoder)


if __name__ == '__main__':
    main()