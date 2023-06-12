import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils import seed_everything
from dataset import load_data
from model import MatrixFactorization
from train import run

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)

    # model args
    #parser.add_argument("--model_name", default="BPRMF", type=str)
    parser.add_argument("--latent_dim", type=int, default=256, help="dimension of latent vector")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=8192, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    data=load_data(args)
    print(args.epochs)
    model=MatrixFactorization(args, data["dataset"])
    run(model, data["train_loader"], data["valid_data"], data["train"], data["dataset"], args)
    
if __name__=="__main__":
    main()