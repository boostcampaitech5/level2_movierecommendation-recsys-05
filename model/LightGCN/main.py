import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils import seed_everything
from dataset import load_data
from model import LightGCN
from train import run, inference

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=8192, help="batch_size")
    
    # model args
    parser.add_argument("--latent_dim", type=int, default=128, help="dimension of latent vector")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--reg", type=float, default=0.001, help="regularization strength")
    parser.add_argument("--n_layers", type=int, default=1, help="numer of layers")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    data = load_data(args)
    train_loader,  dataset = data["train_loader"], data["dataset"]
    print("data prepared")
    model = LightGCN(dataset.train_users_idx, dataset.train_items_idx, dataset.n_users, dataset.n_items, args).to(args.device)
    run(model, train_loader, dataset, args)
    inference(dataset, args)
    
if __name__=="__main__":
    main()