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
    d = x.shape[1]
    xn = x.norm(dim=1)

    num = ( (1 + xn).log() - (1 - xn).log() ).pow(d-1)
    denom = (1 - xn.pow(2)) * xn.pow(d-1) * v.prod(dim=1).sqrt() * np.float_power(2*np.pi, d/2)

    return 2 * num / denom

def monte_integrate(y, d):
    if d == 1:
        area = 2
    elif d == 2:
        area = np.pi

    return area / y.shape[0] * y.sum()

if __name__ == '__main__':
    n = 50
    d = 2

    # cartension product
    x = torch.linspace(-1, 1, n)
    x = torch.cartesian_prod(x, x)
    x = x[x.norm(dim=1) < 1]

    # random parameters for gaussian ball
    mu = 2*torch.rand(1, d) - 1
    v = torch.ones(1, d).abs()
    v[0, 0] = v[0, 1]
    print(f'std: {v.sqrt().numpy()}, mu: {mu.numpy()}')

    y = gauss_ball(x, mu, v)

    # print max point
    print(f'Max: {y.max().numpy()} at {x[y.argmax()].numpy()}')

    # monte carlo integration
    integral = monte_integrate(y, d)
    print(f'Integral: {integral:.4f}')

    if x.shape[1] == 2:
        # plot gaussian ball using 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # x : [n, 2], y : [n]
        ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap='twilight_shifted', edgecolor='none')
        plt.show()
    elif x.shape[1] == 1:
        # plot gaussian ball using 2d plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x[:, 0], y, s=3)
        ax.axis('equal')
        plt.show()

