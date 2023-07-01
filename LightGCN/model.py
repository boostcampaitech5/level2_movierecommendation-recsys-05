import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
from scipy.sparse import dok_matrix
from utils import mat_to_tensor

import numpy as np



class LightGCN(nn.Module):
    def __init__(self, train_u_idx, train_i_idx, n_users, n_items, args):
        super(LightGCN, self).__init__()
        self.train_u_idx=train_u_idx
        self.train_i_idx=train_i_idx
        self.n_users=n_users
        self.n_items=n_items
        self.latent_dim=args.latent_dim
        self.n_layers=args.n_layers
        self.reg=args.reg
        self.init_weight()
        self.graph=self.get_norm_adj_matrix().to(args.device)
        
    def init_weight(self):
        self.user_embedding=nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding=nn.Embedding(self.n_items, self.latent_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def get_norm_adj_matrix(self):
        R=dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        R[self.train_u_idx, self.train_i_idx]=1.0
        R=R.tolil()
        
        adj=dok_matrix((self.n_users+self.n_items, self.n_users+self.n_items), dtype=np.float32)
        adj=adj.tolil()
        adj[:self.n_users, self.n_users:]=R
        adj[self.n_users:, :self.n_users]=R.T
        adj=adj.todok()
        
        nonzeros=np.array(adj.sum(axis=1)) 
        d_mat=np.power(nonzeros, -0.5).flatten()
        d_mat[np.isinf(d_mat)]=0.0
        d_mat=scipy.sparse.diags(d_mat)
        
        norm_adj=d_mat.dot(adj).dot(d_mat).tocsr()
        
        return mat_to_tensor(norm_adj)
        
        
    def forward(self, user, pos_i, neg_i):
        
        '''''''''propagating'''''''''
        all_emb=torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        emb_list=[all_emb]
        for _ in range(self.n_layers):
            all_emb=torch.sparse.mm(self.graph, all_emb)
            emb_list.append(all_emb)
        all_emb=torch.stack(emb_list, dim=1)
        mean_emb=torch.mean(all_emb, dim=1)
        user_final, item_final=torch.split(mean_emb, [self.n_users, self.n_items])
        self.user_final_embedding=nn.Parameter(user_final)
        self.item_final_embedding=nn.Parameter(item_final)
        
        '''''''''bpr_loss'''''''''
        user_e=user_final[user]
        pos_e=item_final[pos_i]
        neg_e=item_final[neg_i]
        
        user_init=self.user_embedding.weight[user]
        pos_init=self.item_embedding.weight[pos_i]
        neg_init=self.item_embedding.weight[neg_i]
        
        pos=torch.sum(torch.mul(user_e, pos_e), dim=1)
        neg=torch.sum(torch.mul(user_e, neg_e), dim=1)
        loss=torch.mean(nn.functional.softplus(neg-pos))   #-logsigmoid(x)=softplus(-x)
        
        reg_term=self.reg*(user_init.norm(2).pow(2)+pos_init.norm(2).pow(2)+neg_init.norm(2).pow(2))/float(len(user))
       
        return loss+reg_term