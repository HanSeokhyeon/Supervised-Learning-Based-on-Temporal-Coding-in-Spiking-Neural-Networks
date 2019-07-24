from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
from data import load_xor_time
import math

k = 100
learning_rate = 0.1
epochs = 1000
train_num = 4

x, y = load_xor_time()

dim_x = 2
dim_h = 4
dim_out = 2

dtype_float = torch.FloatTensor

z0 = Variable(torch.randn(dim_x), requires_grad=False)

W0 = Variable(torch.randn(dim_x, dim_h), requires_grad=True)

z1 = Variable(torch.randn(dim_h), requires_grad=False)

W1 = Variable(torch.randn(dim_h, dim_out), requires_grad=True)

z2 = Variable(torch.randn(dim_out), requires_grad=False)


def model(x, W0, W1):
    z0 = torch.exp(x)
    z1 = spike_layer(z0, W0)
    z2 = spike_layer(z1, W1)

    return z2


def spike_layer(z, W):
    N = W.size()[1]
    z_next = Variable(torch.randn(N), requires_grad=False)
    for i in range(N):
        C = get_causal_set(z, W[:, i])
        if C.size()[0]:
            z_next[i] = W[:, C].sum() * z[C] / (W[:, C].sum() - 1)
        else:
            z_next[i] = math.inf
    return z_next


def get_causal_set(zr_1, wr):
    sort_indices = torch.argsort(zr_1)
    z_sorted = zr_1[sort_indices]
    w_sorted = wr[sort_indices]

    N = zr_1.size()[0]
    for i in range(N):
        if i == N-1:
            next_input_spike = math.inf
        else:
            next_input_spike = z_sorted[i+1]

        first_cond = w_sorted.sum() > 1
        second_cond = w_sorted.sum() * z_sorted.sum() / (w_sorted.sum() - 1) < next_input_spike
        if first_cond != second_cond:
            return sort_indices[:i]
    return torch.Tensor([])


for epoch in range(epochs):
    for i in range(train_num):
        y_out = model(x[i], W0, W1)
        t = y[i].type(torch.ByteTensor)
        loss = -torch.log(torch.exp(-y_out[t.item()]) / torch.exp(-y_out).sum())
        loss.backward()

        W0.data -= learning_rate * W0.grad.data
        W1.data -= learning_rate * W1.grad.data

        print("%d, %f" % (epoch, 1 + loss))

W0.grad.data.zero_()
W1.grad.data.zero_()

y_test_out = model(x, W0, b0, W1, b1).data.numpy()

print(np.round(y_test_out * 1000)/1000)
print(y.numpy())
