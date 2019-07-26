from data import load_xor_time
from SNN import SNN
import matplotlib.pyplot as plt
import numpy as np

x, y = load_xor_time()

snn = SNN(x, y)
snn.gradient_descent()

np.savetxt("loss.csv", snn.loss_data, delimiter=',')

plt.plot(snn.loss_data)
plt.ylim(0, 1)
plt.show()

pass