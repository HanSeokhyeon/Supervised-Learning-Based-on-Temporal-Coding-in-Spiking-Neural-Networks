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

z0 = Variable(torch.randn(dim_x))
W0 = Variable(torch.randn(dim_h, dim_x))
z1 = Variable(torch.randn(dim_h))
W1 = Variable(torch.randn(dim_out, dim_h))
z2 = Variable(torch.randn(dim_out))


def model(x, W0, W1):
    z0 = torch.exp(x)
    z1 = spike_layer(z0, W0)
    z2 = spike_layer(z1, W1)

    return z2


def spike_layer(z, W):
    N = W.size()[0]
    z_next = Variable(torch.zeros(N))
    for i in range(N):
        C = get_causal_set(z, W[i, :])
        if C.size()[0]:
            z_next[i] = W[i, C].sum() * z[C].sum() / (W[i, C].sum() - 1)
        else:
            z_next[i] = 3.4028235 * np.power(10, 38)
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
        sum = w_sorted.sum()
        first_cond = sum > 1
        second_cond = w_sorted.sum() * z_sorted.sum() / (w_sorted.sum() - 1) < next_input_spike
        if first_cond != second_cond:
            return sort_indices[:i+1]
    return torch.Tensor([])


for epoch in range(epochs):
    for i in range(train_num):
        zL = Variable(model(x[i], W0, W1), requires_grad=True)

        g = y[i].type(torch.ByteTensor).item()
        softmax = torch.exp(-zL[g]) / torch.exp(-zL).sum()
        loss = -torch.log(softmax)
        loss.backward(zL)

        W0.data -= learning_rate * W0.grad.data
        W1.data -= learning_rate * W1.grad.data

        print("%d, %f" % (epoch, 1 + loss))

W0.grad.data.zero_()
W1.grad.data.zero_()

y_test_out = model(x, W0, b0, W1, b1).data.numpy()

print(np.round(y_test_out * 1000)/1000)
print(y.numpy())
