from Loss import *
from Config import *
import torch
cfg = Config()
criterion = Loss(cfg)
output = torch.zeros([2,7,7,30])
output[0,0,0,0:5] = torch.tensor([
    1,1,0.01,0.04,1
])
output[0,0,0,5:10] = torch.tensor([
    0,0,0,0,1
])
output[0,0,0,10] = torch.tensor([
    1
])
output[1,4,4,0:5] = torch.tensor([
    1,1,0.01,0.04,1
])
output[1,4,4,5:10] = torch.tensor([
    0,0,0,0,1
])
output[1,4,4,10] = torch.tensor([
    1
])
target = torch.zeros([2,8])
target[0] = torch.tensor([1,1,0.01,0.04,0,0,0,0])
target[1] = torch.tensor([1,1,0.01,0.04,4,4,0,1])
with torch.no_grad():
    criterion.cpu()
    print(criterion(output,target))