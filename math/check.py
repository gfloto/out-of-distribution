import torch
from functools import partial

def g(xn):
    return (1 + xn).log() - (1 - xn).log()

def phi(x):
    assert len(x.shape) == 2

    xn = x.norm(dim=1)[..., None]
    return g(xn) / xn * x

# ------------

def s1(x):
    xn = x.norm(dim=1)
    return 1 / (1 - xn.pow(2))

def s1_score(x):
    xn = x.norm(dim=1)[..., None]
    return 2*x / (1 - xn.pow(2))

# ------------

def s2(x):
    d = x.shape[1]
    xn = x.norm(dim=1)
    return xn.pow(d-1)

def s2_score(x):
    d = x.shape[1]
    xn = x.norm(dim=1)[..., None]
    return (d-1) * x / xn.pow(2)

# ------------

def s3(x):
    d = x.shape[1]
    xn = x.norm(dim=1)
    return g(xn).pow(d-1)

def s3_score(x):
    d = x.shape[1]
    xn = x.norm(dim=1)[..., None]
    return (d-1) * 2*x / (xn * (1 - xn.pow(2)) * g(xn))

# ------------

def s4(x, mu, v):
    px = phi(x)

    diff = px - mu
    f = torch.einsum('ij,ij->i', diff, diff)
    return (-0.5/v * f).exp() 

def s4_score(x, mu, v):
    xn = x.norm(dim=1)[..., None]

    a = x * (phi(x) - mu) / xn.pow(2)
    b = 2 / (1 - xn.pow(2)) - g(xn) / xn
    c = g(xn) / xn * (phi(x) - mu)

    sum_k = (a * b).sum(dim=1)[..., None]
    return -1/v * (x * sum_k  + c)

# ------------
def finite_difference(f, x, h=1e-8):
    assert len(x.shape) == 2

    # fd score
    fd = torch.empty_like(x)
    for i in range(x.shape[1]):
        dx = torch.zeros_like(x)
        dx[:,i] = h

        y1 = f(x).log()
        y2 = f(x + dx).log()

        fd[:,i] = (y2 - y1) / h
    return fd

if __name__ == '__main__':
    n = 100
    d = 2

    mu = 2*torch.rand(d) - 1
    v = torch.randn(1).abs()

    x = torch.rand(n, d) * 2 - 1
    x = x[x.norm(dim=1) < 1]
    x = x.type(torch.float64)

    score = s4_score(x, mu, v)
    fd = partial(s4, mu=mu, v=v)
    fd_score = finite_difference(fd, x)
    print(fd_score.sum(), score.sum())

    # average error
    print((fd_score - score).abs().mean())