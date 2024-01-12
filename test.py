import torch
from torch import Tensor

import numpy as np
from numpy import array

a = Tensor([0, 0, 1, 1]).bool()
print(a)

print(torch.where(a)[0])

print("aa", torch.tensor(10))