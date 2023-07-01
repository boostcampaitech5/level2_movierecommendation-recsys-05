import torch
import os
import numpy as np
import random

def recall_at_10(answer, toplist):
    return len(set(answer) & set(toplist)) / len(answer)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True