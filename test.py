import numpy as np
import torch

a = torch.tensor([[1, 2, 3], [3, 2, 1]])
b = torch.tensor([[3, 2, 1], [1, 2, 3]])
c = a.numpy()
d = b.numpy()

print(np.equal(a,b))
