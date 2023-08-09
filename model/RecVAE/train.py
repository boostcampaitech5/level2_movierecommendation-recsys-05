import pandas as pd
import numpy as np
import os

import torch
from torch.optim import Adam

from utils import recall_at_10


def train(model, opts, data_loader, dataset, epochs, dropout_rate, args):
    model.train()
    total_loss = 0
    for _ in range(epochs) :
        for user in data_loader:
            
            for optimizer in opts:
                optimizer.zero_grad()
                
            mat = dataset.make_matrix(user).to(args.device)
            _, loss = model(mat, beta = args.beta, gamma = args.gamma, dropout_rate = dropout_rate)
            
            total_loss += loss.item()
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()
                
                
def run(model, data_loader, dataset, args):
    best_recall = 0.0
    enc_optimizer=Adam(model.encoder.parameters(), lr=args.lr)
    dec_optimizer=Adam(model.decoder.parameters(), lr=args.lr)
    
    print("Start train...")
    for epoch in range(args.epochs):
        
        if args.not_alter:  #update at once
            train(model, [enc_optimizer, dec_optimizer], data_loader, dataset, 1, 0.5, args)
            
        else:  #alternating update
            train(model, [enc_optimizer], data_loader, dataset, args.enc_epochs, 0.5, args)
            model.update_prior()
            train(model, [dec_optimizer], data_loader, dataset, args.dec_epochs, 0, args)
            
        model.eval()
        recall = 0
    
        with torch.no_grad():
            for user in data_loader:
                mat = dataset.make_matrix(user).to(args.device)
                
                output = model(mat, args.beta, args.gamma, 0.5, calculate_loss=False)
                output[mat == 1] = -np.inf
                
                _, idx = torch.topk(output, k=10)
                pred_items = idx.to("cpu").detach().numpy()
                for u, items in zip(user, pred_items):
                    recall += recall_at_10(dataset.valid[u.item()], items)

        if recall > best_recall:
            best_recall = recall
            if not os.path.exists(args.save_dir_path):
                os.makedirs(args.save_dir_path)
            torch.save(model, f"{args.save_dir_path}/best_model.pt")
            torch.save(model.state_dict(), f"{args.save_dir_path}/state_dict.pt")
            
        print(f"epoch : {epoch} recall : {recall/dataset.n_users}")    


def inference(data_loader, dataset, args):
    
    model = torch.load(f"{args.save_dir_path}/best_model.pt") 
    model.load_state_dict(torch.load(f"{args.save_dir_path}/state_dict.pt"))
    model.to(args.device)
    model.eval()
    topk = []  

    print("Start Inference...")
    with torch.no_grad():
        for user in data_loader:
            mat = dataset.make_matrix(user, train=False).to(args.device)

            output = model(mat, args.beta, args.gamma, 0.5, calculate_loss=False)
            output[mat == 1] = -np.inf
            values, idx = torch.topk(output, k=10)
            pred_items = idx.to("cpu").detach().numpy()
            values = values.to("cpu").detach().numpy()
            
            for u, items, v in zip(user, pred_items, values):
                for item, score in zip(items, v):
                    topk.append((dataset.user_map[u.item()], dataset.item_map[item], score))
                    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        pd.DataFrame(topk, columns = ["user", "item", "score"]).to_csv(f"{args.output_dir}/recvae_submission_addscore_moduletest.csv", index = False)