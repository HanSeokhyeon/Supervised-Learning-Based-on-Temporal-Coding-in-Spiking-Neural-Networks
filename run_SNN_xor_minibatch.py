from data import load_xor
from SNN_xor import SNNMinibatch

data = load_xor()

snn = SNNMinibatch([2, 4, 2])
snn.SGD(data, 1000, 1, 0.1)
