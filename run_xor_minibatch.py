from data import load_xor
from SNN import SNN_minibatch

data = load_xor()

snn = SNN_minibatch([2, 4, 2])
snn.SGD(data, 1000, 1, 0.1)
