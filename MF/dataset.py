import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os

import torch
from torch.utils.data import Dataset, DataLoader

class MFDataset(Dataset):
    def __init__(self, datadict):
        self.data=datadict
        users, items=[], []
        for u, v in self.data.items():
            users+=[u]*len(v)
            items.extend(v)
            
        self.users, self.items = torch.tensor(users), torch.tensor(items)
        
    def __getitem__(self, index):
        return self.users[index], self.items[index]
    
    def __len__(self):
        return self.users.shape[0]



class Preprocessing():
    def __init__(self, data):
        self.data=data
        self.data["user_id"], self.user_map=pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map=pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        
        self.user_negative_itemset=self.get_user_negitemset()
        self.train, self.valid=self.split()
    
    def get_user_negitemset(self):
        user_negative_itemset=defaultdict(list)
        self.user_itemset=self.data.groupby("user_id")["item_id"].apply(list).to_dict()   #user별 시청한 전체 itemid list(train+valid)-> user:itemlist
        for u in self.user_itemset:
            user_negative_itemset[u]=list(set(self.all_items)-set(self.user_itemset[u]))      #user별 시청하지 않은 전체itemid list(-train-valid)-> user:itemlist
        return user_negative_itemset
    
    def negative_sampling(self, user):
        nega_user, nega_item=[], []
        for u in user:
            nega_item+=np.random.choice(self.user_negative_itemset[u], 3, replace = False).tolist()   #+self.valid[u]를 포함? 
            nega_user+=[u]*3
            
        return torch.tensor(nega_user), torch.tensor(nega_item)
            
    def split(self):
        train={}
        valid={}
        for u, v in self.user_itemset.items():
            valitem=np.random.choice(v[int(len(v)*0.4):], 10, replace = False).tolist()
            trainitem=list(set(v)-set(valitem))
            train[u]=trainitem
            valid[u]=valitem
        return train, valid

    
def load_data(args):
    rating_df=pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    rating_df=rating_df.sort_values(["user", "time"])
    dataset=Preprocessing(rating_df)
    train, valid=dataset.split()
    train_data=MFDataset(train)
    train_loader=DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    data={
        "train_loader":train_loader,
        "valid_data":valid,
        "dataset":dataset,
        "train":train
    }
    
    return data