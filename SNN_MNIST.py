from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math
import random
import numpy as np


class SNN:

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [torch.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        self.causal_sets = [[] for i in range(self.num_layers - 1)]
        self.Z = []
        self.Z.append(torch.exp(x.type(torch.FloatTensor)))
        for i, (w, c) in enumerate(zip(self.weights, self.causal_sets)):
            self.Z.append(self.spike_layer(self.Z[i], w, c))

        self.Z[-1].requires_grad_(True)

        return Variable(self.Z[-1], requires_grad=True)

    # forward pass in spiking networks
    # input: z: vector of input spike times
    # input: W: weight matrices. W[i, j] is the weight from neuron j in layer l - 1 to neuron i in layer l
    # output: z_next: Vector of first spike times of neurons
    def spike_layer(self, z, w, c):
        N = w.size()[0]
        z_next = torch.zeros(N)
        for i in range(N):
            tmp_c = self.get_causal_set(z, w[i, :])
            c.append(tmp_c)
            if tmp_c.size()[0]:
                z_next[i] = torch.mul(w[i, tmp_c], z[tmp_c]).sum() / (w[i, tmp_c].sum() - 1)
            else:
                z_next[i] = torch.tensor(1000000000000000000)
        return z_next

    # gets indices of input spikes influencing first spike time of output neuron
    # input: zr_1: Vector of input spike times of length N
    # input: wr: Weight vector of the input spikes
    # Output: C: Causal index set
    @staticmethod
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
            w_sum = w_sorted[:i+1].sum()
            z_out = torch.mul(w_sorted[:i+1], z_sorted[:i+1]).sum() / (w_sorted[:i+1].sum() - 1)
            first_cond = w_sum > 1
            second_cond = z_out < next_input_spike
            if first_cond != second_cond:
                return sort_indices[:i+1]
        return torch.Tensor([])

    def SGD(self, training_data, testing_data, epochs, eta, K, max_norm):
        self.n = len(training_data)
        for j in range(epochs):
            # random.shuffle(training_data)
            for x_mini_batch, y_mini_batch in training_data:
                self.update_mini_batch(x_mini_batch, y_mini_batch, eta, K, max_norm)

            print("Epoch: %d\tloss: %.8f\t" % (j, self.loss_average), end='')
            self.evaluate(training_data)

    def update_mini_batch(self, x_mini_batch, y_mini_batch, eta, K, max_norm):
        nabla_weights = [torch.zeros(w.size())for w in self.weights]
        self.loss_average = 0
        for x, y in zip(x_mini_batch, y_mini_batch):
            x = x.view(784)
            x_binary = (x > torch.tensor([0.5])).float() * 1.79
            delta_nabla_w = self.backprop(x_binary, y, K, max_norm)
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_w)]
            self.loss_average += self.loss.item()/len(x_mini_batch)
        self.weights = [w-(eta/len(x_mini_batch))*nw for w, nw in zip(self.weights, nabla_weights)]

    def backprop(self, x, y, K, max_norm):
        nabla_w = [torch.zeros(w.size()) for w in self.weights]

        # feedforward
        self.zL = self.feedforward(x)

        # backward
        delta = self.get_delta(y).view(self.sizes[-1], 1)
        delta_wsc = self.get_delta_weight_sum_cost(K)

        for l in range(1, self.num_layers):
            nabla_w_z = self.get_nabla_W(self.Z[-l], self.Z[-(l+1)], self.weights[-l], self.causal_sets[-l])
            nabla_w[-l] = torch.mul(delta, nabla_w_z) + delta_wsc[-l]
            nabla_z_z = self.get_nabla_z(self.weights[-l], self.causal_sets[-l])
            delta = torch.mul(delta, nabla_z_z)

        nabla_w = self.normalize_gradient(nabla_w, max_norm)

        return nabla_w

    def get_delta(self, y):
        g = y # label
        softmax = torch.exp(-self.zL-torch.max(-self.zL)) / (torch.exp(-self.zL-torch.max(-self.zL)).sum())
        y_one_hot = F.one_hot(g, num_classes=self.sizes[-1]).type(torch.FloatTensor)

        self.loss = -torch.log(softmax[g])

        # backward pass
        delta = softmax - y_one_hot

        return delta

    def get_delta_weight_sum_cost(self, K):
        delta_wsc = []

        for i in range(self.num_layers-1):
            input_weight = Variable(self.weights[i], requires_grad=True)
            input_weight_sum = 1 - torch.sum(input_weight, dim=1)
            input_weight_sum = torch.cat([torch.zeros(input_weight_sum.size()).view(-1, 1),
                                          input_weight_sum.view(-1, 1)],
                                          dim=1)
            input_weight_sum, _ = input_weight_sum.max(dim=1)
            input_weight_sum = input_weight_sum.sum()

            weight_sum_cost = K * input_weight_sum

            weight_sum_cost.backward()
            delta_wsc.append(input_weight.grad.data)

        return delta_wsc

    def normalize_gradient(self, n_w, max_norm):
        for i in range(self.num_layers-1):
            row_Frobenius_norm = torch.norm(n_w[i], 2, dim=1)# / self.sizes[i+1]
            for j, norm in enumerate(row_Frobenius_norm):
                if norm > max_norm:
                    n_w[i][j] *= max_norm / norm

        return n_w

    @staticmethod
    def get_nabla_W(z_out, z_p, w_p, c):
        nabla_W = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_W[i, j] = (z_p[j]-z_out[i]) / (w_p[i, c_tensor].sum() - 1)

        return nabla_W

    @staticmethod
    def get_nabla_z(w_p, c):
        nabla_z = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_z[i, j] = w_p[i, j] / (w_p[i, c_tensor].sum() - 1)

        return nabla_z

    def evaluate(self, eval_data):
        eval_data = np.array(list(map(lambda x: x[0] + [x[1]], eval_data)))
        eval_x = eval_data[:, :-1]
        eval_y = eval_data[:, -1]
        pred = []
        for x in eval_x:
            pred.append(torch.argmin(self.feedforward(x)).item())
        eval_result = np.equal(pred, eval_y)
        pred = list(map(str, pred))
        eval_y = list(map(str, eval_y))
        print("pred: %s\tlabel: %s\t%d / %d" % (' '.join(pred), ' '.join(eval_y), np.sum(eval_result), self.n))

        return