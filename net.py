import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x): # x.size() : 4, 2
        z = torch.exp(x) # 4, 2
        z = self.spiking_layer(z, self.fc1)
        z = self.spiking_layer(z, self.fc2)
        return z

    def spiking_layer(self, zr_1, fc):
        N = fc.out_features
        zr = torch.zeros(N) # 4, 2
        for i in range(N):
            causal_set = self.get_causal_set(zr_1, fc.weight[i, :])
            if causal_set.size()[0]:
                zr[i] = fc.weight[i, causal_set].sum() * zr_1[causal_set].sum() / (fc.weight[i, causal_set].sum()-1)
            else:
                zr[i] = math.inf
        return zr

    def get_causal_set(self, z, w):
        N = z.size()[0]
        sort_indices = torch.argsort(z, dim=0)
        z_sorted = z[sort_indices]
        w_sorted = w[sort_indices]
        for i in range(N):
            if i == N:
                next_input_spike = math.inf
            else:
                next_input_spike = z_sorted[i]
            first_cond = w_sorted.sum() > 1
            tmp = w_sorted.sum()*z_sorted.sum()/(w_sorted.sum()-1)
            second_cond = tmp < next_input_spike
            if first_cond != second_cond:
                return sort_indices[:i-1]
        return torch.Tensor([])
