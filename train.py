import os
import torch
import numpy as np
from tqdm import tqdm

from utils import ptnp
from plot import save_images, save_gen

# adv model options
@torch.no_grad()
def gen_adv_nograd(x_train, adv_model):
    return gen_adv(x_train, adv_model)

def gen_adv_grad(x_train, adv_model):
    return gen_adv(x_train, adv_model)

def gen_adv(x_train, adv_model):
    bs = x_train.shape[0]
    device = x_train.device

    # generate adv data to pass with training data
    # delta is beta distribution with alpha=1, beta=5
    delta = torch.distributions.beta.Beta(
        torch.tensor([1.]), torch.tensor([5.])
    ).sample((bs,)).squeeze().to(device)

    hard = torch.rand(bs).to(device) # how hard to determine if gen or not

    # generate adversarial sample
    noise = torch.randn_like(x_train).to(device)
    noise = torch.nn.functional.avg_pool2d(noise, 3, stride=1, padding=1)
    #x_input = torch.cat((x_train, noise), dim=1)
    x_gen = adv_model(noise, delta, hard)

    return x_gen, delta, hard

# model options
@torch.no_grad()
def model_nograd(model, x):
    return model(x, encode_only=True)

def model_grad(model, x):
    return model(x)

# main training loop
def train(model, adv_model, loader, loss_fn, optim_model, optim_adv, args):
    model.train()
    recon_track, iso_track, label_track = [], [], []
    diff_track, adv_label_track = [], []
    for i, (x_train, _) in enumerate(tqdm(loader)):
        #if i >= 100: break
        x_train = x_train.to(args.device)

        # step 1: regular model training     
        if i % 2 == 0:
            optim_model.zero_grad

            x_train = x_train[:32]
            bs = x_train.shape[0]

            # get adversarial sample
            x_gen, _, _ = gen_adv_grad(x_train, adv_model)

            # input to model
            x = torch.cat((x_train, x_gen), dim=0)
            label = torch.cat((torch.zeros(bs), torch.ones(bs))).to(args.device)
            
            # push through model
            _, mu, x_out = model_grad(model, x)
            recon, iso, adv = loss_fn(x, x_out, mu, label)
            loss = args.recon_lambda * recon + iso + adv
            #loss = adv
                
            # optimize and clip gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim_model.step()

            # save loss for plotting 
            recon_track.append(ptnp(recon))
            iso_track.append(ptnp(iso))
            label_track.append(ptnp(adv))

        # step 2: adversarial model training
        else:
            bs = x_train.shape[0]
            optim_adv.zero_grad()

            # generate adversarial sample
            x_gen, delta, hard = gen_adv_grad(x_train, adv_model)

            # push through model
            _, mu, x_out = model_grad(model, x_gen)

            # lpips difference loss
            #percept = loss_fn.x_dist.model
            #lpips_diff = percept(x_train, x_gen)
            #l1_diff = (x_train - x_gen).abs().mean()
            #diff = lpips_diff + l1_diff
            #diff_loss = (delta - diff).square().mean()



            # label loss
            # optimal encoder: length-1 always 1, meaning hard: 0 should predict a 1
            label = torch.ones(bs).to(args.device)
            length = loss_fn.adv_loss(mu, None, return_length=True)
            label_loss = loss_fn.BCE(length, hard).mean()

            #loss = 0.1*diff_loss + label_loss
            loss = label_loss

            # optimize and clip gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adv_model.parameters(), args.max_grad_norm)
            optim_adv.step()

            # save loss for plotting
            diff_track.append(ptnp(diff_loss))
            adv_label_track.append(ptnp(label_loss))

    # save sample images
    if args.save_images:
        save_images(x, x_out, os.path.join('results', args.test_name, 'sample.png'))
        save_gen(x_gen, delta, hard, os.path.join('results', args.test_name, 'gen.png'))

    return np.mean(recon_track), np.mean(iso_track), np.mean(label_track), np.mean(diff_track), np.mean(adv_label_track) 

if __name__== '__main__':
    pass
