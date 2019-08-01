from data import load_MNIST_data
from SNN_MNIST import SNN

train_loader, test_loader = load_MNIST_data()

snn = SNN([784, 800, 10])
snn.SGD(train_loader, test_loader, epochs=100, eta=0.1, K=100, max_norm=100)

pass