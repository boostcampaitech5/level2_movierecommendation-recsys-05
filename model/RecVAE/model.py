import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy


def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar+np.log(2*np.pi)+(x-mu).pow(2)/logvar.exp())


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, eps=0.1):
        super(Encoder, self).__init__()
        
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.ln1=nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.ln2=nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3=nn.Linear(hidden_dim, hidden_dim)
        self.ln3=nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4=nn.Linear(hidden_dim, hidden_dim)
        self.ln4=nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5=nn.Linear(hidden_dim, hidden_dim)
        self.ln5=nn.LayerNorm(hidden_dim, eps=eps)
        
        self.fc_mu=nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar=nn.Linear(hidden_dim, latent_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, dropout_rate):
        norm=x.pow(2).sum(dim=-1).sqrt()  #variance
        x=x/norm[:, None]
        
        x=F.dropout(x, p=dropout_rate, training=self.training)
        
        h1=self.ln1(self.swish(self.fc1(x)))
        h2=self.ln2(self.swish(self.fc2(h1)+h1))
        h3=self.ln3(self.swish(self.fc3(h2)+h1+h2))
        h4=self.ln4(self.swish(self.fc4(h3)+h1+h2+h3))
        h5=self.ln5(self.swish(self.fc5(h4)+h1+h2+h3+h4))
        
        return self.fc_mu(h5), self.fc_logvar(h5)
        
    def swish(self, x):
        return x.mul(torch.sigmoid(x))


class CompositePrior(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, mixture_w=[3/20, 3/4, 1/10]):  #weights for composite prior
        super(CompositePrior, self).__init__()
        
        self.mixture_w=mixture_w
        
        self.mu_prior=nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior=nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_normal_prior=nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_normal_prior.data.fill_(10)
        
        self.encoder_prev=Encoder(input_dim, hidden_dim, latent_dim)
        self.encoder_prev.requires_grad_(False)
                        
    def forward(self, x, z):
        post_mu, post_logvar=self.encoder_prev(x, dropout_rate=0)
        
        stnd_prior=log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior=log_norm_pdf(z, post_mu, post_logvar)
        norm_prior=log_norm_pdf(z, self.mu_prior, self.logvar_normal_prior)
        
        gaussians=[g.add(np.log(w)) for g, w in zip([stnd_prior, post_prior, norm_prior], self.mixture_w)]
        
        return torch.logsumexp(torch.stack(gaussians, dim=-1), dim=-1)  #(stndprior, post_prior, norm_prior)
    
    
class RecVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RecVAE, self).__init__()
        self.encoder=Encoder(input_dim, hidden_dim, latent_dim)
        self.prior=CompositePrior(input_dim, hidden_dim, latent_dim)
        self.decoder=nn.Linear(latent_dim, input_dim)
        
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)
                
    def reparameterize(self, mu, logvar):
        if self.training:
            std=torch.exp(0.5*logvar)
            eps=torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def update_prior(self):
        self.prior.encoder_prev.load_state_dict(deepcopy(self.encoder.state_dict()))
        
    def forward(self, rating, beta, gamma, dropout_rate, calculate_loss=True):
        mu, logvar=self.encoder(rating, dropout_rate)
        z=self.reparameterize(mu, logvar)
        x_pred=self.decoder(z)
        
        if calculate_loss:
            if gamma:  #new approach for setting beta
                kl_weight=rating.sum(dim=-1)*gamma
            elif beta:
                kl_weight=beta
            mll=(F.log_softmax(x_pred, dim=-1)*rating).sum(dim=-1).mean()
            kld=(log_norm_pdf(z, mu, logvar)-self.prior(rating, z)).sum(dim=-1).mul(kl_weight).mean()
            
            nega_elbo=-(mll-kld)
            return (mll, kld), nega_elbo
        else:
            return x_pred