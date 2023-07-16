import torch
import numpy as np

# lazy is good
def ptnp(x):
    return x.detach().cpu().numpy()