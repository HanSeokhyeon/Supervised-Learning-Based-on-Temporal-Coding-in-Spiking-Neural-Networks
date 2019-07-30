from torch.autograd import Variable
import torch
import math
import random
import numpy as np


class SNN:

    def __init__(self, x, y):
        # hyperparameter
        self.k = 10
        self.learning_rate = 0.1
        self.epochs = 100
        self.train_num = 4

        # model size
        self.dim_x = 2
        self.dim_h = 4
        self.dim_out = 2

        # model variable
        self.z0 = Variable(torch.randn(self.dim_x))
        self.W0 = Variable(torch.randn(self.dim_h, self.dim_x))

        self.z1 = Variable(torch.randn(self.dim_h))

        self.W1 = Variable(torch.randn(self.dim_out, self.dim_h))

        self.z2 = Variable(torch.randn(self.dim_out))

        # result
        self.loss_data = []

        # input data
        self.x = x
        self.y = y

    # model graph
    def model(self, x):
        # initialize causal sets
        self.C1 = []
        self.C2 = []

        # model
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
            tmp_C = self.get_causal_set(z, W[i, :])
            if tmp_C.size()[0]:
                z_next[i] = W[i, tmp_C].sum() * z[tmp_C].sum() / (W[i, tmp_C].sum() - 1)
                C.append(tmp_C)
            else:
                z_next[i] = torch.pow(torch.tensor(10), 25)
        return z_next

    # gets indices of input spikes influencing first spike time of output neuron
    # input: zr_1: Vector of input spike times of length N
    # input: wr: Weight vector of the input spikes
    # Output: C: Causal index set
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
            loss_sum = 0
            for i in range(self.train_num):
                nabla_loss_W0, nabla_loss_W1 = self.backprop(self.x[i], self.y[i])
                self.W1 = self.update(self.W1, nabla_loss_W1)
                self.W0 = self.update(self.W0, nabla_loss_W0)

                prediction = self.evaluate()

                zL_str = list(map(lambda x: "inf" if x > 1e10 else str(x.item()), self.zL))
                print("Epoch: %d\tloss: %f\tzL[0]: %s\tzL[1]: %s\tprediction: %d\tlabel: %d" % (epoch, self.loss, zL_str[0][:5], zL_str[1][:5], prediction, self.y[i].item()))

                loss_sum += self.loss.item()
            self.loss_data.append(loss_sum)

    def update(self, w, nabla):
        w -= self.learning_rate * nabla

        return w

    def backprop(self, x, y):

        # feedforward
        self.zL = Variable(self.model(x), requires_grad=True)

        g = y.type(torch.ByteTensor).item() # label
        softmax = torch.exp(-self.zL[g]) / torch.exp(-self.zL).sum()
        self.loss = -torch.log(softmax)


        # backward pass
        self.loss.backward(self.zL)
        delta = self.zL.grad.data

        tmp_zL = Variable(self.model(x), requires_grad=True)

        tmp_g = y.type(torch.ByteTensor).item()  # label
        tmp_softmax = torch.exp(-tmp_zL[tmp_g]) / torch.exp(-tmp_zL).sum()
        tmp_loss = -torch.log(tmp_softmax)

        tmp_delta = torch.autograd.grad(tmp_loss, tmp_zL)[0]

        nabla_W1 = self.get_nabla_W(self.z2, self.z1, self.W1, self.C2)
        nabla_W0 = self.get_nabla_W(self.z1, self.z0, self.W0, self.C1)

        nabla_z1 = self.get_nabla_z(self.W1, self.C2)

        nabla_loss_W1 = torch.mul(delta, nabla_W1.transpose(0, 1)).transpose(0, 1)
        nabla_loss_W0 = torch.mul(torch.mul(delta, nabla_z1.transpose(0, 1)).transpose(0, 1), nabla_W0.transpose(0, 1)).transpose(0, 1)

        return nabla_loss_W0, nabla_loss_W1

    def get_nabla_W(self, z_out, z_p, w_p, c):
        nabla_W = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_W[i, j] = (z_p[j]-z_out[i]) / (w_p[i, :].sum() - 1)

        return nabla_W

    def get_nabla_z(self, w_p, c):
        nabla_z = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_z[i, j] = w_p[i, j] / (w_p[i, :].sum() - 1)

        return nabla_z

    def evaluate(self):
        pred = torch.argmin(self.zL).item()
        return pred


class SNN_minibatch:

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [torch.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        self.causal_sets = [[] for i in range(self.num_layers - 1)]
        self.Z = []
        self.Z.append(torch.exp(torch.tensor(x).type(torch.FloatTensor)))
        for i, (w, c) in enumerate(zip(self.weights, self.causal_sets)):
            self.Z.append(self.spike_layer(self.Z[i], w, c))

        self.Z[-1].requires_grad_(True)

        return Variable(self.Z[-1], requires_grad=True)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        self.n = len(training_data)
        for j in range(epochs):
            # random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, self.n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch %d: %d / %d" % (j, self.evaluate(test_data), n_test))
            else:
                print("Epoch: %d\tloss: %.8f\t" % (j, self.loss_average), end='')
                self.evaluate(training_data)

    def update_mini_batch(self, mini_batch, eta):
        nabla_weights = [torch.zeros(w.size())for w in self.weights]
        self.loss_average = 0
        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_w)]
            self.loss_average += self.loss.item()/len(mini_batch)
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_weights)]

    def backprop(self, x, y):
        nabla_w = [torch.zeros(w.size()) for w in self.weights]

        # feedforward
        self.zL = self.feedforward(x)
        delta = self.get_delta(y)

        for l in range(1, self.num_layers):
            nabla_w_l = self.get_nabla_W(self.Z[-l], self.Z[-(l+1)], self.weights[-l], self.causal_sets[-l])
            nabla_w[-l] = torch.mul(delta, nabla_w_l.transpose(0, 1)).transpose(0, 1)
            nabla_z_l = self.get_nabla_z(self.weights[-l], self.causal_sets[-l])
            delta = torch.mul(delta, nabla_z_l.transpose(0, 1)).transpose(0, 1)

        return nabla_w

    def get_delta(self, y):
        g = y # label
        softmax = torch.exp(-self.zL[g]) / torch.exp(-self.zL).sum()
        self.loss = -torch.log(softmax)

        # backward pass
        self.loss.backward(self.zL)
        delta = self.zL.grad.data

        return delta

    # forward pass in spiking networks
    # input: z: vector of input spike times
    # input: W: weight matrices. W[i, j] is the weight from neuron j in layer l - 1 to neuron i in layer l
    # output: z_next: Vector of first spike times of neurons
    def spike_layer(self, z, w, c):
        N = w.size()[0]
        z_next = torch.zeros(N)
        for i in range(N):
            tmp_c = self.get_causal_set(z, w[i, :])
            if tmp_c.size()[0]:
                z_next[i] = w[i, tmp_c].sum() * z[tmp_c].sum() / (w[i, tmp_c].sum() - 1)
                c.append(tmp_c)
            else:
                z_next[i] = torch.pow(torch.tensor(10), 25)
        return z_next

    # gets indices of input spikes influencing first spike time of output neuron
    # input: zr_1: Vector of input spike times of length N
    # input: wr: Weight vector of the input spikes
    # Output: C: Causal index set
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

    def get_nabla_W(self, z_out, z_p, w_p, c):
        nabla_W = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_W[i, j] = (z_p[j]-z_out[i]) / (w_p[i, :].sum() - 1)

        return nabla_W

    def get_nabla_z(self, w_p, c):
        nabla_z = torch.zeros(w_p.size())
        for i, c_tensor in enumerate(c):
            for j in c_tensor.numpy():
                nabla_z[i, j] = w_p[i, j] / (w_p[i, :].sum() - 1)

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
