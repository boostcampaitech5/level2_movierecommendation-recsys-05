import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os

import torch
from torch.utils.data import Dataset, DataLoader

class VAEDataSet(Dataset):
    def __init__(self, dataset):
        self.users = dataset.all_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx): 
        user = self.users[idx]
        return torch.LongTensor([user])



class Preprocessing():
    def __init__(self, data):
        self.data=data
        
        self.data["user_id"], self.user_map = pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map = pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()
        self.train, self.valid, self.submission = self.split()
    
            
    def split(self):
        train = {}
        valid = {}
        submission = {}
        for u, v in self.user_itemset.items():
            submission[u] = v
            valitem = v[-2:]  #가장 최근의 시청 이력
            valitem += np.random.choice(v[int(len(v)*0.2):-2], 8, replace = False).tolist()  #일정 기간 이후의 시청 이력에서 random sampling
            trainitem = list(set(v) - set(valitem))
            train[u] = trainitem
            valid[u] = valitem
            
        return train, valid, submission
    
    
    def make_matrix(self, user_list, train = True):
        mat = torch.zeros(size = (user_list.size(0), self.n_items))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, self.train[user.item()]] = 1
            else:
                mat[idx, self.submission[user.item()]] = 1
        return mat
    
    
def load_data(args):
    rating_df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    rating_df = rating_df.sort_values(["user", "time"])
    dataset = Preprocessing(rating_df)
    trainset = VAEDataSet(dataset)
    train_loader=DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    sub_loader=DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
    data={
        "dataset" : dataset,
        "train_loader" : train_loader,
        "sub_loader" : sub_loader,
    }
    
    return data