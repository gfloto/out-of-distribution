import torch 
from tqdm import tqdm
from einops import rearrange

def f(t, T):
    # f(t) = cos(pi*t/T + s) / 2*(1 + s)**2
    s = torch.tensor(0.008)
    return torch.cos( torch.pi/2 * (t/T + s) / (1 + s) ).square()

def make_alpha_bars(T):
    # alpha bar = f(t) / f(0)
    x = torch.arange(T)
    alpha_bars = f(x, T) / f(0, T)
    return alpha_bars

def make_alphas(T):
    # beta = 1 - (alpha_bar_t / alpha_bar_{t-1})
    alpha_bars = make_alpha_bars(T)
    alpha = alpha_bars[1:] / alpha_bars[:-1]
    alpha = torch.cat([torch.tensor([1]), alpha])
    return alpha

class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.device = device
        self.alpha_bars = make_alpha_bars(T).to(device)
        self.alphas = make_alphas(T).to(device)
        check_alpha = self.alphas.cumprod(dim=0)

        # location of diffusion
        self.gammas = 1 - self.alphas.sqrt()
        self.k = torch.tensor(0).to(device)

    # get xt given x0 and t for training the diffusion model
    def q_xt_x0(self, x0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        xt += (1 - alpha_bar.sqrt()) * self.k

        return xt, eps
    
    # get x_{t-1} given x_t and t for likelihood calculation
    def q_xt_xtm1(self, xtm1, t):
        alpha = self.alphas[t]
        eps = torch.randn_like(xtm1)

        xt = alpha.sqrt() * xtm1 + (1 - alpha).sqrt() * eps
        xt += (1 - alpha.sqrt()) * self.k

        return xt, eps

    def p_x0_xt(self, xt, pred, t):
        alpha_bar = self.alpha_bars[t]
        return (xt - (1 - alpha_bar).sqrt() * pred) / alpha_bar.sqrt()
    
    # get x_{t-1} given x_t and t for sampling from the diffusion model
    def p_xtm1_xt(self, xt, pred, t):
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t-1]
        
        c1 = alpha_bar_prev.sqrt() * (1 - alpha) / (1 - alpha_bar)
        c2 = alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
        #c3 = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha)
        c3 = (1 - alpha)

        eps = torch.randn_like(xt)
        xtm1 = c1*pred + c2*xt + c3.sqrt()*eps

        return xtm1
    
    # for T steps, get x_T, x_{T-1}, ..., x_0 to generate a sample
    @torch.no_grad()
    def sample(self, model, autoenc, dim):
        x = torch.randn(64, dim).to(self.device) + self.k
        for t_ in tqdm(range(self.T-1, 0, -1)):
            t = torch.tensor([t_]).float().to(self.device) / self.T

            pred = model(x, t)
            x = self.p_xtm1_xt(x, pred, t_)

            # print average norm of x
            #print(x.norm(dim=1).mean())
        
        # normalize
        x = autoenc.decode(x)
        return x

    # given some x0, get the likelihood (ignoring the constant terms, and L_0 for now...)
    # E[-log p(x0)] <= sum L_t-1 = sum_{t=2}^T D_KL q(x_t-1 | x_t) || p(x_t-1 | x_t)
    # L_t-1 (x0) = below
    def likelihood(self, x0, model):
        total_ll = torch.zeros(x0.shape[0]).to(self.device)
        for t_ in range(self.T-1, 0, -10):
            t = torch.tensor([t_]).float().to(self.device) / self.T
            xt, eps = self.q_xt_x0(x0, t_)

            pred = model(xt, t)
            ll = (pred - x0).square().mean(dim=(1))
            total_ll += ll
        return total_ll

import matplotlib.pyplot as plt
if __name__ == '__main__':
    T = 2000
    t = torch.arange(T)
    alpha_bars = make_alpha_bars(T)
    alpha = make_alphas(T)
    check = alpha.cumprod(dim=0)

    plt.plot(t/T, alpha_bars)
    plt.plot(t/T, check)
    plt.show()
