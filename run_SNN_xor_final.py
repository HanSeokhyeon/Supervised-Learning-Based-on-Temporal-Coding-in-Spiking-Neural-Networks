from data import load_xor
from SNN_xor import SNNWeightCostSum

data = load_xor()

snn = SNNWeightCostSum([2, 4, 2])
snn.SGD(data, epochs=100, mini_batch_size=1, eta=0.1, K=100, max_norm=100)

pass
