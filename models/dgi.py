import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseline_models import LogReg, GCN


class Discriminator(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.layer = nn.Bilinear(out_dim, out_dim, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h1, h2):
        c_x = c.expand_as(h1)
        sc1 = self.layer(h1, c_x).t()
        sc2 = self.layer(h2, c_x).t()
        logits = torch.cat((sc1, sc2), 1)
        return logits

class Readout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        return torch.mean(h, 1, keepdim=True)

class DGI(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers,  dropout_rate, gnn_type, alpha , beta, add_self_loops):
        super().__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim, n_layers, dropout_rate, gnn_type, alpha , beta, add_self_loops)
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def get_embedding(self, data):
        h1 = self.encoder(data.x, data.edge_index)
        return h1.detach()

    def forward(self, data, cfeat):
        h1 = self.encoder(data.x, data.edge_index) # shared encoder
        h2 = self.encoder(cfeat, data.edge_index) # shared encoder
        c = self.sigm(self.read(h1))
        ret = self.disc(c, h1, h2)
        return ret