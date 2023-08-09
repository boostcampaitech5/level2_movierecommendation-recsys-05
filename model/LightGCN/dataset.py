import pandas as pd
import numpy as np
from collections import defaultdict
import os

from scipy.sparse import dok_matrix
from utils import mat_to_tensor

import torch
from torch.utils.data import Dataset, DataLoader


class BPRDataset(Dataset):
    def __init__(self, dataset):
        users = []
        pos_items = []
        neg_items = []
        for u in dataset.all_users:  #use all
            for i in dataset.train[u]: #negative sampling - 2
                users += [u]*2
                pos_items += [i]*2
                neg_items += np.random.choice(dataset.user_negative_itemset[u], 2, replace = False).tolist()

        self.users, self.pos, self.nega = torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)
        
    def __getitem__(self, index):
        return self.users[index], self.pos[index], self.nega[index]
    
    def __len__(self):
        return self.users.shape[0]
    
    
class Preprocessing():
    def __init__(self, data):
        self.data=data
        
        self.data["user_id"], self.user_map = pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map = pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        
        self.user_negative_itemset = self.get_user_negitemset()
        self.train, self.valid = self.split()
        self.train_mask, self.sub_mask = self.generate_mask()
        
    
    def get_user_negitemset(self):
        user_negative_itemset = defaultdict(list)
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()   #user별 시청한 전체 itemid list(train+valid)
        for u in self.user_itemset:
            user_negative_itemset[u] = list(set(self.all_items)-set(self.user_itemset[u]))      #user별 시청하지 않은 전체 itemid list
        return user_negative_itemset
    
    
    def split(self):
        train = {}
        valid = {}
        
        for u, v in self.user_itemset.items():
            valitem = np.random.choice(v[int(len(v)*0.4):], 10, replace = False).tolist()
            trainitem = list(set(v)-set(valitem))
            train[u] = trainitem
            valid[u] = valitem
                
        return train, valid
    
    def generate_mask(self):
        
        train_users, train_items = [], []
        sub_users, sub_items = [], []
        
        for u, v in self.train.items():
            train_users += [u]*len(v)
            train_items.extend(v)
            
        self.train_users_idx=train_users
        self.train_items_idx=train_items
        
        for u, v in self.user_itemset.items():
            sub_users += [u]*len(v)
            sub_items.extend(v)
             
        train_R = dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        train_R[train_users, train_items] = 1.0
        train_mask = mat_to_tensor(train_R).to_dense()*(-np.inf)
        train_mask = torch.nan_to_num(train_mask, nan=0.0)
        
        sub_R = dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        sub_R[sub_users, sub_items] = 1.0
        sub_mask = mat_to_tensor(sub_R).to_dense()*(-np.inf)
        sub_mask = torch.nan_to_num(sub_mask, nan=0.0)

        return train_mask, sub_mask

def load_data(args):
    rating_df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    rating_df = rating_df.sort_values(["user", "time"])
    dataset = Preprocessing(rating_df)
    train_data=BPRDataset(dataset)
    train_loader=DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    data={
        
        "dataset" : dataset,
        "train_loader" : train_loader,
        "valid" : dataset.valid
    }
    
    return data