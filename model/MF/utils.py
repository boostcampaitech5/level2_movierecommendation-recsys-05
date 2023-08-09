import torch.nn as nn
import torch
import os
import numpy as np
import random

def recall_at_10(answer, toplist):
    return len(set(answer) & set(toplist)) / len(answer)

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos, neg):
        return -torch.log(self.gamma+torch.sigmoid(pos-neg)).mean()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True