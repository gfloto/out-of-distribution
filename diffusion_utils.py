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
    alpha_bars = make_alpha_bars(T+1)
    alpha = alpha_bars[1:] / alpha_bars[:-1]
    return alpha

class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.device = device
        self.alpha_bars = make_alpha_bars(T).to(device)
        self.alphas = make_alphas(T).to(device)

        check_alpha = self.alphas.cumprod(dim=0)

    # get xt given x0 and t for training the diffusion model
    def q_xt_x0(self, x0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        return xt, eps
    
    # get x_{t-1} given x_t and t for likelihood calculation
    def q_xt_xtm1(self, xtm1, t):
        alpha = self.alphas[t]
        eps = torch.randn_like(xtm1)

        xt = alpha.sqrt() * xtm1 + (1 - alpha).sqrt() * eps
        return xt, eps
    
    # get x_{t-1} given x_t and t for sampling from the diffusion model
    def p_xtm1_xt(self, xt, pred, t):
        alpha_bar = self.alpha_bars[t]
        alpha = self.alphas[t]
        sig = (1 - alpha_bar).sqrt()

        eps = torch.randn_like(xt)
        c = (1 - alpha) / (1 - alpha_bar).sqrt()
        xtm1 = 1/alpha.sqrt() * (xt - c*pred) + sig*eps

        return xtm1
    
    # for T steps, get x_T, x_{T-1}, ..., x_0 to generate a sample
    @torch.no_grad()
    def sample(self, model, autoenc, dim):
        x = torch.randn(16, dim, dim).to(self.device)
        for t_ in tqdm(range(self.T-1, 0, -1)):
            t = torch.tensor([t_]).float().to(self.device) / self.T

            pred = model(x, t)
            x = self.p_xtm1_xt(x, pred, t_)

            # get average length of x
            temp = rearrange(x, 'b h w -> b (h w)')
            avg_len = temp.square().sum(dim=1).sqrt().mean()
            print(f'avg length: {avg_len:.5f}')

        x = rearrange(x, 'b h w -> b (h w)')
        x = autoenc.decode(x)
        return x

    # given some x0, get the likelihood (ignoring the constant terms, and L_0 for now...)
    # E[-log p(x0)] <= sum L_t-1 = sum_{t=2}^T D_KL q(x_t-1 | x_t) || p(x_t-1 | x_t)
    # L_t-1 (x0) = below
    def likelihood(self, x0, model):
        total_ll = 0
        for t in range(self.T):
            # use an estimate of the expectation with a single sample
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            sig = (1 - alpha_bar).sqrt()
            c = (1 - alpha)**2 / (2*sig * alpha * (1 - alpha_bar))

            xtm1, eps = self.q_xt_xtm1(x0, t)
            pred = model(xtm1, t)
            
            ll = c * (eps - pred).square().sum(dim=(1,2,3))
            total_ll = total_ll + ll

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
