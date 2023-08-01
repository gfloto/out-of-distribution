import torch 

def f(t, T):
    # f(t) = cos(pi*t/T + s) / 2*(1 + s)**2
    s = torch.tensor(0.008)
    return torch.cos( torch.pi/2 * (t/T + s) / (1 + s) ).square()

def make_alpha_bars(T):
    # alpha bar = f(t) / f(0)
    x = torch.arange(T)
    alpha_bars = f(x, T) / f(0, T)
    return alpha_bars

def make_alphas(T=1000):
    # beta = 1 - (alpha_bar_t / alpha_bar_{t-1})
    alpha_bars = make_alpha_bars(T+1)
    alpha = alpha_bars[1:] / alpha_bars[:-1]
    return alpha

class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.alpha_bars = make_alpha_bars(T).to(device)
        self.alphas = make_alphas(T).to(device)

    # get xt given x0 and t for training the diffusion model
    def forward(self, x0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        return xt, eps
    
    # get x_{t-1} given x_t and t for sampling from the diffusion model
    def sample(self, xt, pred, t):
        alpha_bar = self.alpha_bars[t]
        alpha = self.alphas[t]
        sig = (1 - alpha_bar).sqrt()
        eps = torch.randn_like(xt)

        c = (1 - alpha) / (1 - alpha_bar).sqrt()
        xtm1 = 1/alpha.sqrt() * (xt - c*pred) + sig*eps

        return xtm1


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
