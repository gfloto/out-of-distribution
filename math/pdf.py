import torch 
from torch.func import jacrev, vmap
import numpy as np 
from functools import partial
from scipy.special import gamma
import matplotlib.pyplot as plt

def g(xn):
    return (1 + xn).log() - (1 - xn).log()

def phi(x):
    assert len(x.shape) == 2

    xn = x.norm(dim=1)[..., None]
    return g(xn) / xn * x

def monte_integrate(y, d):
    vol = np.pi**(d/2) / gamma(d/2 + 1)

    return vol / y.shape[0] * y.sum()

def gauss_ball(x, mu, v):
    assert len(x.shape) == 2
    assert len(mu.shape) == 1
    assert len(v.shape) == 1

    xn = x.norm(dim=1)
    c = norm_const(xn, v, d=x.shape[1])
    px = phi(x)

    # implement batched dot product
    diff = px - mu
    f = torch.einsum('ij,ij->i', diff, diff)
    e = (-0.5 * f / v).exp() 

    return c * e 

def norm_const(xn, v, d):
    assert len(xn.shape) == 1
    assert len(v.shape) == 1

    num = g(xn).pow(d-1)
    denom = (1 - xn.pow(2)) * xn.pow(d-1) * v.sqrt().pow(d) * np.float_power(2*np.pi, d/2)
    return 2 * num / denom

    #a = g(xn).pow(d) / xn.pow(d)
    #b = 2*xn / (g(xn) * (1 - xn.pow(2))) - 1
    #c = v.sqrt().pow(d) * np.float_power(2*np.pi, d/2)

    #return a * b / c

def gauss_ball_score(x, mu, v):
    assert len(x.shape) == 2
    assert len(mu.shape) == 1
    assert len(v.shape) == 1

    d = x.shape[1]
    xn = x.norm(dim=1)[..., None]

    a = x * (phi(x) - mu) / xn.pow(2)
    b = 2 / (1 - xn.pow(2)) - g(xn) / xn
    c = g(xn) / xn * (phi(x) - mu)

    sum_k = (a * b).sum(dim=1)[..., None]
    s4 = -1/v * (x * sum_k  + c)

    s1 = 2*x / (1 - xn.pow(2))
    s2 = -(d-1) * x / xn.pow(2) 
    s3 = (d-1) * 2*x / (xn * (1 - xn.pow(2)) * g(xn))

    return s1 + s2 + s3 + s4

def finite_difference(f, x, h=1e-8):
    assert len(x.shape) == 2

    # finite difference
    fd = torch.empty_like(x)
    for i in range(x.shape[1]):
        dx = torch.zeros_like(x)
        dx[:,i] = h

        y1 = f(x).log()
        y2 = f(x + dx).log()

        fd[:,i] = (y2 - y1) / h
    return fd

def log_pdf(x, mu, v):
    return gauss_ball(x, mu, v).log()

if __name__ == '__main__':
    n = 1000
    d = 1

    # cube discard method, poor scaling performance...
    x = torch.randn(n, d, dtype=torch.float64)
    while True:
        x = x[x.norm(dim=1) < 1]
        if x.shape[0] >= n: break

        x_add = torch.rand(n, d, dtype=torch.float64)
        x = torch.cat([x, x_add], dim=0)
    x = x[:n]
    print('good')

    # random parameters for gaussian ball
    mu = 2*torch.rand(d) - 1
    v = torch.rand(1).abs()
    print(f'v: {v.numpy()}, mu: {mu.numpy()}')

    # get pdf
    y = gauss_ball(x, mu, v)
    y_score = gauss_ball_score(x, mu, v)

    # get finite difference approximation of score
    gb = partial(gauss_ball, mu=mu, v=v)
    y_score_fd = finite_difference(gb, x)

    # print max point
    print(f'Max: {y.max().numpy()} at {x[y.argmax()].numpy()}')

    # monte carlo integration
    integral = monte_integrate(y, d)
    print(f'Integral: {integral:.4f}')
    print(f'Average score difference: {(y_score - y_score_fd).abs().mean().numpy()}')

    if x.shape[1] == 2:
        # plot gaussian ball using 3d plot
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(x[:, 0], x[:, 1], y, cmap='twilight_shifted', edgecolor='none')
        ax1.set_title('Gaussian Ball PDF')

        ax2 = fig.add_subplot(122)
        diff = (y_score-y_score_fd).abs()
        ax2.scatter(x[:, 0], y_score[:, 0], s=3, c='darkblue')
        ax2.scatter(x[:, 1], y_score[:, 1], s=3, c='deepskyblue')

        ax2.scatter(x[:, 0], y_score_fd[:, 0], s=3, c='forestgreen')
        ax2.scatter(x[:, 1], y_score_fd[:, 1], s=3, c='limegreen')
        ax2.set_yscale('symlog')
        ax2.set_title('Gaussian Ball Score')
        plt.show()
    elif x.shape[1] == 1:
        # plot gaussian ball using 2d plot
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.scatter(x[:, 0], y, s=3)
        ax1.axis('equal')
        ax1.set_title('Gaussian Ball PDF')

        ax2 = fig.add_subplot(122)
        ax2.scatter(x[:, 0], y_score[:,0], s=3, c='deepskyblue')
        ax2.scatter(x[:, 0], y_score_fd[:,0], s=3, c='limegreen')
        ax2.set_yscale('symlog')
        ax2.set_title('Gaussian Ball Score Differerence')
        plt.show()

