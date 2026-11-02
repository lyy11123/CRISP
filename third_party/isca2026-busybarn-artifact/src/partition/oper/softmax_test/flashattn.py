"""Reference FlashAttention-style softmax experiment helpers."""

import torch

torch.manual_seed(456)

N, d = 16, 8
Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
expected_attention = expected_softmax @ V_mat

Br = 4
Bc = d

O = torch.zeros((N, d))

for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br
    Qi = Q_mat[block_start_Br:block_end_Br, :]
    Oi = torch.zeros((Br, d))
    li = torch.zeros((Br, 1))
    mi = torch.full((Br, 1), -torch.inf)

    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc

        Kj = K_mat[block_start_Bc:block_end_Bc, :]
        Vj = V_mat[block_start_Bc:block_end_Bc, :]

        Sij = Qi @ Kj.T
        mi_new = torch.max(torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
        Pij_hat = torch.exp(Sij - mi_new)
        li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]
        Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj

        mi = mi_new

    Oi = Oi / li

    O[block_start_Br:block_end_Br, :] = Oi


assert torch.allclose(O, expected_attention)
