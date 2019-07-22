import data
import net
import torch
import torch.optim as optim
import torch.nn as nn

# 1. load data
data = data.load_xor_data()
datasize = data[1].size()[0]

# 2. define a neural net
net = net.Net(2, 4, 2)

# 3. define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train the neural net
for epoch in range(100000):

    running_loss = 0.0

    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameters gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / datasize))

print('Finished Training')

# 5. test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(outputs.data.numpy(), predicted.numpy(), labels.numpy())
