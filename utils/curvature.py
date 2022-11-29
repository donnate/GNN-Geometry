import numpy as np
import scipy as sc
import torch


def augmented_forman(A, gamma=1):
    # gamma regulates the contribution of the triangle
    if not torch.is_tensor(A):
        A = torch.Tensor(A)
    d = torch.sum(A, dim=1)
    n = A.shape[0]
    d_temp = np.repeat(d, n)
    D = d_temp.reshape(n, n)
    d_mod = np.where(d == 0, 1, d)
    tr = torch.mm(torch.t(A), A)
    tr = tr.fill_diagonal_(0)
    F_temp = 4 - D - torch.t(D) + 3 * gamma * tr
    F_sum = torch.mm(F_temp, A)
    F = torch.diagonal(F_sum, 0) / d_mod
    return (F)

def pre_curvature(embedding, data):
    num_edge = data.edge_index.shape[1]
    num_nodes = data.x.shape[0]
    temp = embedding
    dist = sc.spatial.distance.cdist(temp.detach().numpy(),
                                     temp.detach().numpy())
    # calculate curvature
    value, _ = torch.topk(-1 * torch.Tensor(list(dist[np.triu_indices(num_nodes)])),
                          int(num_edge / 2 + num_nodes))
    A_temp = torch.Tensor(dist) <= -1 * value[int(num_edge / 2 + num_nodes) - 1]
    A = A_temp.to(torch.float32)
    if not torch.is_tensor(A):
        A = torch.Tensor(A)
    ind = np.diag_indices(A.shape[0])
    A[ind[0], ind[1]] = torch.zeros(A.shape[0])
    return A
