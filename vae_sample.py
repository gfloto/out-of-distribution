import os, sys 
import json
import argparse
import torch 
from einops import rearrange 

from plot import save_sample
from models import get_model

# get args for testing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='damage-2', help='path to model checkpoint')
    parser.add_argument('--save_name', type=str, default='model.pt', help='name of test folder')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='device type')

    args = parser.parse_args()

    # update paths
    args.model_load_path = os.path.join('results', args.test_name, args.save_name)

    # check inputs
    if not os.path.exists(args.model_load_path):
        raise ValueError('model_path does not exist')

    return args

@ torch.no_grad()
def sample(model, tilt, spheres, batch_size, device='cuda'):
    assert batch_size % 4 == 0 # TODO: this is the number of angles...
    assert batch_size % spheres == 0
    model.eval()

    # generate 8 images at random, vary the angle condition to see effect on reconstruction
    z = torch.randn(batch_size // 4, spheres, model.lat_dim // spheres).to(device)
    z = tilt / z.norm(dim=-1, keepdim=True) * z
    z = rearrange(z, 'b s d -> b (s d)')
    z = torch.cat([z for _ in range(4)], dim=0)

    ones = torch.ones(batch_size // 4).to(device)
    angle = torch.cat([i * ones for i in range(4)], dim=0)

    x_out = model.decode(z, angle)
    x_out = torch.clamp(x_out, 0, 1)

    # save sample
    save_sample(x_out, os.path.join('results', load_args.test_name, 'gen_sample.png'))

if __name__ == '__main__':
    load_args = get_args()

    # load .json from test_name
    with open(os.path.join('results', load_args.test_name, 'args.json'), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # load model
    model = get_model(args.model_yaml_path, args.lat_dim, args.use_timestep).to(args.device)
    state = torch.load(load_args.model_load_path, map_location=args.device)
    model.load_state_dict(state)

    # sample from model
    sample(model, args.tilt, args.spheres, load_args.batch_size, device=load_args.device)
