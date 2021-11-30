
import tvm

from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import time

# Load pre-trained model

from mnist_example import ReguNet, IrreguNet

model = ReguNet() # As you can see, here I load the regular net
# Here I add the weights to the model from, chosen_net is the irregular one that I stored
# -> see mnist_example.py, line 278
model.load_state_dict(torch.load("normal_model_state_dict"))
model.eval()

input_shape = [1, 1, 28, 28]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Load a test image
mnist_path = ""
mnist_data = MNIST(root = mnist_path, transform=transforms.ToTensor(), download=True, train = False)
start_time = time.time()
img = np.array(mnist_data.data[0].reshape(1,1,28,28))

# Import the graph to Relay

input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)

# Relay Build

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)

# Execute the portable graph on TVM

from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
whole_time = 0
for j in range(10):
    for i in range(10000):
        img = np.array(mnist_data.data[i].reshape(1,1,28,28))
        m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
        # Execute
        start_time = time.time()
        m.run()
        # Get outputs
        tvm_output = m.get_output(0)
        whole_time += time.time()-start_time
print("Needed time: ", whole_time)

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.numpy()[0])

print("Done")
