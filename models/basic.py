import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(dim)

        self.t_emb = nn.Sequential(
            nn.Linear(1, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
        )

        self.proj = nn.Sequential( 
            nn.Linear(2*dim, dim),
            nn.ELU(),
        )

        self.block1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
        )

        self.block2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
        )


    def forward(self, x_og, t):
        t_emb = self.t_emb(t)[None, ...]
        t_emb = torch.repeat_interleave(t_emb, repeats=x_og.shape[0], dim=0)
        x = torch.cat((x_og, t_emb), dim=-1)
        x = self.proj(x)

        x_ = x + x_og

        x = self.block1(self.norm(x_))
        x = self.block2(self.norm(x))
        return x + x_og

class Basic(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.n_layers = 12

        self.first = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ELU(),
        )

        self.blocks = nn.ModuleList([Block(hidden_dim) for _ in range(self.n_layers)])

        self.last = nn.Sequential(
            nn.Linear(hidden_dim, dim),
        )


    def forward(self, x, t):
        x = self.first(x)
        for block in self.blocks:
            x = block(x, t)
        x = self.last(x)
        return x
