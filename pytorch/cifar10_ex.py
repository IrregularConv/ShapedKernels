import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# from net import ResNet9
# from mnist_example import IrreguConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

class BrainDamage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, mask, stride, padding):
        if torch.cuda.is_available():
            mask = mask.to(input.device)
        if mask.size() == weights.size():
            weights = weights * mask
        output = F.conv2d(input, weights, stride = stride, padding = padding)
        ctx.save_for_backward(input, weights, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w, mask = ctx.saved_variables
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_output) # TODO: include padding and stride
        if ctx.needs_input_grad[1]:
            w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_output)
            w_grad *= mask
        return x_grad, w_grad, None, None, None

class IrreguConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, mask = None):
        super(IrreguConv, self).__init__(in_channels, out_channels, kernel_size)
        
        if mask == None or mask.size() != (out_channels, in_channels, kernel_size, kernel_size):
            self.mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        else:
            self.mask = mask

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Set up weights given in nn.Conv2d
        weights = torch.zeros((out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
        nn.init.kaiming_normal(weights)
        self.weight = torch.nn.Parameter(weights*self.mask)

    def forward(self, input: Tensor) -> Tensor:
        output = BrainDamage.apply(input, self.weight, self.mask, self.stride, self.padding)
        
        return output

class IrreguConvCustomWeights(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, mask = None):
        super(IrreguConvCustomWeights, self).__init__(in_channels, out_channels, kernel_size)
        
        if mask == None or mask.size() != (out_channels, in_channels, kernel_size, kernel_size):
            self.mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        else:
            self.mask = mask

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Set up weights manually and set nn.Conv2d param to None to save memory
        weights = torch.zeros((out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
        nn.init.kaiming_normal(weights)
        self.weights = torch.nn.Parameter(weights*self.mask)
        self.weight = None

    def forward(self, input: Tensor) -> Tensor:
        output = BrainDamage.apply(input, self.weights, self.mask, self.stride, self.padding)
        
        return output


class Net(nn.Module): # Needs 2132.343 s for 20 epochs and gets 64 % accuracy
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class IrregularNet(nn.Module): # Needs
    def __init__(self):
        super().__init__()
        self.conv1 = IrreguConv(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    @staticmethod
    def add_mask(kernel_mask1, kernel_mask2):
        IrregularNet.conv1 = IrreguConv(3, 16, 5, mask = kernel_mask1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


irregular_net = IrregularNet()
kernel_5_5 = 3 * [[[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]]]
mask_8_3_5_5 = torch.tensor(16*[kernel_5_5])
irregular_net.add_mask(mask_8_3_5_5, (torch.rand(size=(32, 8, 5, 5)) < 0.7).int())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(irregular_net.parameters(), lr=0.001, momentum=0.9)


print("Training start")

irregular_l = []
start_time = time.time()
for epoch in range(20):  # loop over the dataset multiple times

    total_loss = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()
        inputs, labels = data
        # forward + backward + optimize
        outputs = irregular_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    irregular_l += [total_loss / i]
print('Finished Training in time ', time.time() -start_time)


regular_net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(regular_net.parameters(), lr=0.001, momentum=0.9)


print("Training start")

regular_l = []
start_time = time.time()
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()
        inputs, labels = data
        # forward + backward + optimize
        outputs = regular_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    regular_l += [total_loss / i]

plt.plot(irregular_l, label = 'irregular loss')
plt.plot(regular_l, label = 'regular loss')

plt.legend()
plt.show()


PATH = './cifar_irregular_net'
torch.save(irregular_net.state_dict(), PATH)

net = IrregularNet()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))