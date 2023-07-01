import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils import seed_everything
from dataset import load_data
from model import RecVAE
from train import run, inference

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
    
    # model args
    parser.add_argument("--hidden_dim", type=int, default=600, help="hidden dimension size")
    parser.add_argument("--latent_dim", type=int, default=200, help="dimension of latent vector")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--enc_epochs", type=int, default=3, help="number of epochs for encoder")
    parser.add_argument("--dec_epochs", type=int, default=1, help="number of epochs for decoder")
    
    parser.add_argument("--gamma", type=float, default=0.004, help="gamma")
    parser.add_argument("--beta", type=int, default=None, help="beta")
    parser.add_argument("--not_alter", type=bool, default=False, help="alternating updates for encoder and decoder")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    data = load_data(args)
    train_loader, sub_loader, dataset = data["train_loader"], data["sub_loader"], data["dataset"]
    model = RecVAE(dataset.n_items, args.hidden_dim, args.latent_dim).to(args.device)
    run(model, train_loader, dataset, args)
    inference(sub_loader, dataset, args)
    
if __name__=="__main__":
    main()