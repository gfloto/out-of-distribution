import torch 
import numpy as np 
import matplotlib.pyplot as plt

def gauss_ball(x, mu, v):
    c = norm_const(x, v)

    px = phi(x)

    # impliment batched dot product
    diff = px - mu
    f = torch.einsum('ij,ij->i', diff, diff / v)
    e = (-0.5 * f).exp() 

    return c * e 

def phi(x):
    xn = x.norm(dim=1)[..., None]
    a = (1 + xn).log() - (1 - xn).log()

    return a/xn * x 

def norm_const(x, v):
    n = x.shape[1]
    xn = x.norm(dim=1)

    d = (1 - xn.pow(2)) *  v.prod(dim=1) * np.float_power(2*np.pi, n/2)

    return 2 / d

if __name__ == '__main__':
    n = 200
    d = 1

    x = torch.linspace(-1, 1, n).view(-1, d)
    v = torch.randn(1, d).abs()
    mu = 2*torch.rand(1, d) - 1

    print(mu, v)

    y = gauss_ball(x, mu, v)
    x = x.squeeze()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_aspect('equal')
    plt.show()

