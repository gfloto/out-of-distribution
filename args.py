import os 
import json
import argparse

import numpy as np

def bool_str(x):
    if x in ['True', 'true']: return True 
    elif x in ['False', 'false']: return False 
    else: raise ValueError('invalid boolean')

# return args for training
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='dev')
    parser.add_argument('--data_path', default='/drive2/ood/', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default=None, help='experiment path to resume training from')

    parser.add_argument('--dataset', default='cifar10', help='dataset, either odo or damage')
    parser.add_argument('--lat_dim', type=int, default=36, help='latent dimension')

    parser.add_argument('--recon', default='l2', help='loss, either l2 or l1')
    parser.add_argument('--recon_lambda', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--noise', type=float, default=0.01, help='noise level')
    parser.add_argument('--norm', default='true', help='normalize data')
    parser.add_argument('--metric', default='l2', help='metric to use for latent space')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

    parser.add_argument('--max_grad_norm', type=float, default=1., help='max grad norm')

    parser.add_argument('--save_images', type=bool, default=True, help='boolean, sample images')
    parser.add_argument('--test_freq', type=int, default=20, help='frequency of saving model')
    parser.add_argument('--device', default='cuda', help='device being used')

    args = parser.parse_args()

    # TODO: change this
    args.use_timestep = True
    args.norm = bool_str(args.norm)

    # latent codes must be a perfect square for diffusion stage
    d = np.sqrt(args.lat_dim)
    if d.is_integer(): d = int(d)
    else: raise ValueError('latent dimension must be a perfect square')

    # asserts
    assert args.test_name is not None, 'enter a test name'
    assert args.lat_dim > 0, 'latent dimension must be positive'
    assert args.recon in ['l1', 'l2'], 'recon loss must be l1 or l2'
    assert args.metric in ['inner_prod', 'l2'], 'metric must be inner_prod or l2'

    # make results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(os.path.join('results', args.test_name)):
        os.makedirs(os.path.join('results', args.test_name))

    # save args
    save_args(args)
    return args

# save args to .json
def save_args(args):
    save_path = os.path.join('results', args.test_name, 'args.json')
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)