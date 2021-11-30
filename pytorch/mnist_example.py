import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim

import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

use_GPU = False


num_filters = 8
input_filter_size = 1
kernel_size = 5
kernel_mask = (torch.rand(size=(num_filters, input_filter_size, kernel_size, kernel_size)) < 0.5).int()

mnist_path = ""
mnist_data = MNIST(root = mnist_path, transform=transforms.ToTensor(), download=True, train = False)

class BrainDamage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, mask, stride, padding):
        if torch.cuda.is_available():
            mask = mask.to(input.device)
        if mask.size() == weights.size():
            weights = weights * mask
        output = F.conv2d(input, weights,stride = stride, padding = padding)
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

class IrreguNet(nn.Module):
    def __init__(self):
        super(IrreguNet, self).__init__()
        self.conv1 = IrreguConv(1, 8, 5, mask = None)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)

    @staticmethod
    def add_mask(kernel_mask):
        IrreguNet.conv1 = IrreguConv(1, 8, 5, mask = kernel_mask)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class IrreguNet2(nn.Module):
    def __init__(self):
        super(IrreguNet2, self).__init__()
        self.conv1 = IrreguConvCustomWeights(1, 8, 5, mask = kernel_mask)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReguNet(nn.Module):

    def __init__(self):
        super(ReguNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

batch_size = 64

def train(epoch, train_data, model, optimizer, net_num = 1):
    batch_size = train_data.batch_size
    #progressbar setup
    pbar = tqdm.tqdm(total=len(train_data), desc=f"train epoch {epoch}.{net_num}")
    
    model.train()

    train_error = 0
    correct_predictions = 0
    
    criterion = F.cross_entropy

        

    for i, (input, target) in enumerate(train_data):
        if torch.cuda.is_available() and use_GPU:
            input = input.float().cuda()
            target = target.cuda()

        output = model(input)

        loss = criterion(output, target)

        module = model.conv1

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        optimizer.step()
       
        train_error += loss.item()

        correct_predictions += (torch.argmax(output, dim=-1) == target).sum().item()
        pbar.update(1)
        pbar.set_postfix({
            "loss": train_error/(i+1),
            "accuracy": correct_predictions/((i+1)*batch_size)
        })
    pbar.close()
    return train_error/(i+1), correct_predictions/((i+1)*batch_size)

def test(epoch, test_data, model):
    batch_size = test_data.batch_size
    #progressbar setup
    pbar = tqdm.tqdm(total=len(test_data), desc=f"test epoch {epoch}")

    model.eval()

    test_error = 0
    correct_predictions = 0
    
    criterion = F.cross_entropy
    for i, (input, target) in enumerate(test_data):
        if torch.cuda.is_available() and use_GPU:
            input = input.float().cuda()
            target = target.cuda()

        output = model(input)

        loss = criterion(output, target)
        
        test_error += loss.item()
        
        correct_predictions += (torch.argmax(output, dim=-1) == target).sum().item()
        
        pbar.update(1)
        pbar.set_postfix({
            "loss": test_error/(i+1),
            "accuracy": correct_predictions/((i+1)*batch_size)
        })

    pbar.close()
    return test_error/(i+1), correct_predictions/((i+1)*batch_size)

mnist_train = MNIST(root = mnist_path, transform=transforms.ToTensor(), download=True, train = True)
mnist_test = MNIST(root = mnist_path, transform=transforms.ToTensor(), download=True, train = False)

test_loader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)

def run_training(epochs, net, net_num = 1):
    if torch.cuda.is_available() and use_GPU:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = 0.001,momentum=0.9)

    train_l = []
    test_l = []
    start_time = time.time()
    for epoch in range(epochs):
        train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

        train_error, train_accuracy = train(epoch, train_loader, net, optimizer, net_num = net_num)
        train_l += [[train_error, train_accuracy]]
        
        test_error, test_accuracy = test(epoch, test_loader, net)
        test_l += [[test_error, test_accuracy]]

    return net, test_l

if __name__ == "__main__":
    # normal_model = ReguNet()
    # net_norm, test_loss = run_training(20, normal_model)

    # torch.save(normal_model.state_dict(), "normal_model_state_dict")

    search_epochs = 3
    random_tryouts = 15

    all_nets = []

    irregular_net = IrreguNet()

    kernel_5_5 = input_filter_size * [[[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]]]
    mask = torch.tensor(num_filters*[kernel_5_5])
    irregular_net.add_mask(mask)

    irregu_model_ret, irregu_test_error = run_training(20, irregular_net)

    regu_net = ReguNet()
    regu_model_ret, regu_test_error = run_training(20, regu_net)

    fig, axs = plt.subplots(1, 2, figsize = (10, 10))
    axs[0].plot([row[0] for row in irregu_test_error], label = 'irregular error')
    axs[0].plot([row[0] for row in regu_test_error], label = 'regular error')

    axs[1].plot([row[1] for row in irregu_test_error], label = 'irregular accuracy')
    axs[1].plot([row[1] for row in regu_test_error], label = 'regular accuracy')
    axs[1].set(ylim=(0.4, 1))

    axs[0].set_title('Errors')
    axs[0].legend()
    axs[1].set_title('Accuracies')
    axs[1].legend()

    plt.show()


    # for i in range(random_tryouts):
    #     random_mask = (torch.rand(size=(num_filters, input_filter_size, kernel_size, kernel_size)) < 0.5).int()
    #     model_search = IrreguNet()
    #     IrreguNet.add_mask(kernel_mask=random_mask)
    #     model_ret, test_error = run_training(search_epochs, model_search, net_num = i)
    #     all_nets += [[model_ret, test_error[-1][0]]]

    # min_error = np.argmin([row[1] for row in all_nets]) # use argmin here

    # chosen_net = all_nets[min_error][0]

    # net_to_rule_them_all, test_loss = run_training(20, chosen_net)
    # torch.save(net_to_rule_them_all.state_dict(), "chosen_net_state_dict")
    # torch.save(net_to_rule_them_all, "chosen_net")


# ----------------------------------------------
# Some other implementations to compare nets
if False:

    irregu_model = IrreguNet((torch.rand(size=(num_filters, input_filter_size, kernel_size, kernel_size)) < 0.5).int())
    irregu_model_custom_w = IrreguNet2()
    regu_model = ReguNet()

    all_nets = [irregu_model, irregu_model_custom_w, regu_model]

    fig, axs = plt.subplots(1, 2, figsize = (10, 10))

    time_l = []

    epochs = 10
    for i, net in enumerate(all_nets):

        if torch.cuda.is_available() and use_GPU:
            net.cuda()
        optimizer = optim.SGD(net.parameters(), lr = 0.001,momentum=0.9)

        train_l = []
        test_l = []
        start_time = time.time()
        for epoch in range(epochs):
            train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

            train_error, train_accuracy = train(epoch, train_loader, net)
            train_l += [[train_error, train_accuracy]]
            
            test_error, test_accuracy = test(epoch, test_loader, irregu_model)
            test_l += [[test_error, test_accuracy]]

        axs[0].plot([row[0] for row in train_l])
        axs[1].plot([row[1] for row in test_l])
        time_l += [time.time()-start_time]

    axs[0].set_title('Errors')
    axs[1].set_title('Accuracies')

    for i, ax in enumerate(axs):
        ax.legend(['Train irregular', 'Test irregular', 'Train irregular custom', 'Test irregular custom', 'Train regular', 'Test regular'])

    print('training_time irregular: ', time_l[0], ' training_time irregular custom', time_l[1], ', training_time regular: ', time_l[2])

    plt.show()


# regu_model = ReguNet()
# if torch.cuda.is_available() and use_GPU:
#     regu_model.cuda()
# optimizer = optim.SGD(regu_model.parameters(), lr = 0.001,momentum=0.9)

# train_l2 = []
# test_l2 = []
# time_2 = time.time()
# for epoch in range(epochs):
#     train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

#     train_error, train_accuracy = train(epoch, train_loader, regu_model)
#     train_l2 += [[train_error, train_accuracy]]
    
#     test_error, test_accuracy = test(epoch, test_loader, regu_model)
#     test_l2 += [[test_error, test_accuracy]]


# for i, ax in enumerate(axs):
#     ax.plot([row[i] for row in train_l1])
#     ax.plot([row[i] for row in test_l1])
#     ax.plot([row[i] for row in train_l2])
#     ax.plot([row[i] for row in test_l2])
#     if i == 0:
#         ax.set_title('Errors')
#     else:
#         ax.set_title('Accuracies')
#     ax.legend(['Train irregular', 'Test irregular', 'Train regular', 'Test regular'])

# plt.show()