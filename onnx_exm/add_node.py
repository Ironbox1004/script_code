import onnx
import onnx.helper as helper
import numpy as np

import torch


model = onnx.load("super_resolution.onnx")

class Preprocess(torch.nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.mean = torch.rand(1,1,1,3)
        self.std = torch.rand(1,1,1,3)

    def forward(self, x):
        x = x.float()
        x = (x/255.0 - self.mean) / self.std
        x = x.permute(0,3,1,2)

        return x

# pre = Preprocess()
#
# torch.onnx.export(
#     pre,
#     (torch.zeros(1,224,224,3,dtype=torch.uint8),),
#     "preprocess.onnx",
#     input_names = ['input'],
#     dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                   }
# )

pre_onnx = onnx.load("preprocess.onnx")

for n in pre_onnx.graph.node:
    n.name = f"pre_{n.name}"
    for i in range(len(n.input)):
        n.input[i] = f"pre_{n.input[i]}"
    for i in range(len(n.output)):
        n.output[i] = f"pre_{n.output[i]}"

for n in model.graph.node:
    if n.name == "Conv_0":
        n.input[0] = "pre/" + pre_onnx.graph.output[0].name


for n in pre_onnx.graph.node:
    model.graph.node.append(n)


input_name = "pre/" + pre_onnx.graph.input[0].name
model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
model.graph.input[0].name = input_name

onnx.save_model(model,"super_resolution_pre.onnx")
