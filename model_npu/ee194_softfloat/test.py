from math import isinf
from pdb import run

import torch
import struct
import numpy as np

hex_vals = []


with open("test.txt", "r") as f:
    for line in f:
        hex = line.strip("\n").removeprefix("0x").strip(" ")
        hex_vals.append(hex)

tensors = []
for h in hex_vals:
    # Pack hex into 2-byte unsigned short and unpack as float16
    raw = struct.pack("H", int(h, 16))
    f16 = np.frombuffer(raw, dtype=np.float16)[0]
    tensors.append(f16)

tensor_batch = torch.tensor(tensors, dtype=torch.float16)
running_sum = 0
for x, tensor in enumerate(tensor_batch):
    print(x, hex_vals[x], tensor)
    # if not tensor.isnan() and not tensor.isinf():
    #     running_sum += tensor
    # else:
    #     print(tensor)
print(running_sum)
print(tensor_batch)  # tensor([1.0000, 1.5000-1.0000], dtype=torch.float16)
