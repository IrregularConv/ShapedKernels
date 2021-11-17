import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from mnist_example import IrreguNet, ReguNet

mnist_path = ""
mnist_test = MNIST(root = mnist_path, transform=transforms.ToTensor(), download=True, train = False)
test_loader = DataLoader(dataset=mnist_test, batch_size=1, shuffle=True)

model = IrreguNet()
model.load_state_dict(torch.load("chosen_net_state_dict"))
model.eval()

test_error = 0
correct_predictions = 0
for i, (input, target) in enumerate(test_loader):
    output = model(input)
    correct_predictions += (torch.argmax(output, dim=-1) == target).sum().item()
        
print("Accuracy ", correct_predictions/(i+1))