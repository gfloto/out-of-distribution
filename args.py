import os 
import json
import argparse

def bool_str(x):
    if x in ['True', 'true']: return True 
    elif x in ['False', 'false']: return False 
    else: raise ValueError('invalid boolean')

# return args for training
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='dev')
    parser.add_argument('--data_path', default='./data', help='train dataset, either odo or damage')
    parser.add_argument('--resume_path', default="dev", help='experiment path to resume training from')

    parser.add_argument('--dataset', default='cifar10', help='dataset, either odo or damage')
    parser.add_argument('--lat_dim', type=int, default=256, help='latent dimension')
    # specify multiple latent dimensions to train over, e.g. --latent_dims 2 4 8 16 32 64 128
    parser.add_argument('--lat_dims', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128], help='latent dimension')
    parser.add_argument('--compute_ci', type=bool, default=True, help='compute confidence intervals for metrics')
    parser.add_argument('--ci_iterpolate_method', default='lowess', help='interpolation method for confidence intervals')
    parser.add_argument('--ci_confidence_levels', type=float, nargs='+', default=[0.01, 0.05, 0.1], help='confidence levels for confidence intervals')

    parser.add_argument('--recon', default='l2', help='loss, either l2 or l1')
    parser.add_argument('--recon_lambda', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--noise', type=float, default=0.1, help='noise level')
    parser.add_argument('--norm', default='true', help='normalize data')
    parser.add_argument('--metric', default='inner_prod', help='metric to use for latent space')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

    parser.add_argument('--max_grad_norm', type=float, default=1., help='max grad norm')

    parser.add_argument('--save_images', type=bool, default=True, help='boolean, sample images')
    parser.add_argument('--test_freq', type=int, default=10, help='frequency of saving model')
    parser.add_argument('--device', default='cuda', help='device being used')

    args = parser.parse_args()

    # TODO: change this
    args.use_timestep = True

    args.norm = bool_str(args.norm)

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