from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
from data import load_xor_time
import math


class SNN:

    def __init__(self, x, y):
        # hyperparameter
        self.k = 10
        self.learning_rate = 0.1
        self.epochs = 1000
        self.train_num = 4

        # model size
        self.dim_x = 2
        self.dim_h = 4
        self.dim_out = 2

        # model variable
        self.z0 = Variable(torch.randn(self.dim_x))
        self.W0 = Variable(torch.randn(self.dim_h, self.dim_x))

        self.z1 = Variable(torch.randn(self.dim_h))
        self.C1 = []
        self.W1 = Variable(torch.randn(self.dim_out, self.dim_h))

        self.z2 = Variable(torch.randn(self.dim_out))
        self.C2 = []

        # input data
        self.x = x
        self.y = y

    # model graph
    def model(self, x):
        self.z0 = torch.exp(x)
        self.z1 = self.spike_layer(self.z0, self.W0, self.C1)
        self.z2 = self.spike_layer(self.z1, self.W1, self.C2)

        return self.z2

    # forward pass in spiking networks
    # input: z: vector of input spike times
    # input: W: weight matrices. W[i, j] is the weight from neuron j in layer l - 1 to neuron i in layer l
    # output: z_next: Vector of first spike times of neurons
    def spike_layer(self, z, W, C):
        N = W.size()[0]
        z_next = torch.zeros(N)
        for i in range(N):
            C.append(self.get_causal_set(z, W[i, :]))
            if C[i].size()[0]:
                z_next[i] = W[i, C[i]].sum() * z[C[i]].sum() / (W[i, C[i]].sum() - 1)
            else:
                z_next[i] = 3.4028235 * np.power(10, 38)
        return z_next

    # gets indices of input spikes influencing first spike time of output neuron
    # input: zr_1: Vector of input spike times of length N
    # input: wr: Weight vector of the input spikes
    # O
    def get_causal_set(self, zr_1, wr):
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

    def gradient_descent(self):
        for epoch in range(self.epochs):
            for i in range(self.train_num):
                self.backprop(x[i], y[i])
                pass

            print("%d" % epoch)

    def update(self):
        pass

    def backprop(self, x, y):
        # initialization
        nabla_W0 = torch.zeros(self.W0.size())
        nabla_W1 = torch.zeros(self.W1.size())

        # feedforward
        zL = Variable(self.model(x), requires_grad=True)

        g = y.type(torch.ByteTensor).item() # label
        softmax = torch.exp(-zL[g]) / torch.exp(-zL).sum()
        loss = -torch.log(softmax)

        # backward pass
        loss.backward(zL)
        delta = zL.grad.data

        for i, c_tensor in enumerate(self.C2):
            for j in c_tensor.numpy():
                nabla_W1[i, j] = (self.z1[j]-self.z2[i]) / (self.W1[i, :].sum() - 1)

        tmp_nabla_W1 = self.get_nabla(self.z2, self.z1, self.W1, self.C2)

        return

    def get_nabla(self, z_out, z_p, w_p, c):
        nabla_W = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_W[i, j] = (z_p[j]-z_out[i]) / (w_p[i, :].sum() - 1)

        return nabla_W


    def evaluate(self):
        pass



x, y = load_xor_time()

snn = SNN(x, y)
snn.gradient_descent()
