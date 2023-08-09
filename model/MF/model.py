import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, args, dataset):
        super(MatrixFactorization, self).__init__()
        self.device=args.device
        self.latent_dim=args.latent_dim
        self.n_users=dataset.n_users
        self.n_items=dataset.n_items
        
        self.user_embedding=nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding=nn.Embedding(self.n_items, self.latent_dim)
        self.linear=nn.Linear(self.latent_dim, 1, bias=False)
        
        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user, item):
        embed_user=self.user_embedding(user)
        embed_item=self.item_embedding(item)
        x=embed_user*embed_item
        return self.linear(x).view(-1)