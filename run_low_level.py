from data import load_xor_time
from SNN import SNN
import matplotlib.pyplot as plt
import numpy as np

x, y = load_xor_time()

for i in range(10):
    snn = SNN(x, y)
    snn.gradient_descent()

    np.savetxt("loss%d.csv" % i, snn.loss_data, delimiter=',')

    plt.plot(snn.loss_data)

    plt.show()

pass