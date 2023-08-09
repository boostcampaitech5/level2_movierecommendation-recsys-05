import torch
import os
import numpy as np
import random

def recall_at_10(valid, top):
    sum_recall = 0.0
    num_users = len(valid)
    for i in range(num_users):
        sum_recall += len(set(valid[i]) & set(top[i])) / float(len(valid[i]))
    return sum_recall / num_users

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def mat_to_tensor(mat):
    coo=mat.tocoo().astype(np.float32)
    value=torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    
    return torch.sparse.FloatTensor(indices, value, coo.shape)