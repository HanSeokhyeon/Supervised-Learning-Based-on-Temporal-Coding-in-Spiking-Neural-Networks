import data
import net
import torch
import torch.optim as optim
import torch.nn as nn
import time

run_with_gpu = True

# 0. check device
use_cuda = run_with_gpu and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 1. load data
train_loader, test_loader = data.load_MNIST_data()

# 2. define a neural net
net = net.Net(784, 800, 10).to(device)

# 3. define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train the neural net
for epoch in range(10):
    # check time
    start = time.time()

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(-1, 784)

        # zero the parameters gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    end = time.time() - start
    print('[%d] loss: %.3f time: %f' % (epoch + 1, running_loss / i, end))

print('Finished Training')

# 5. test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(-1, 784)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Accuracy: %.4f' % (correct/10000))
