import pandas as pd
import numpy as np
import os

import torch
from torch.optim import Adam

from utils import recall_at_10

def run(model, train_loader, dataset, args):
    best_recall = 0
    optimizer = Adam(model.parameters(), lr = args.lr)
    
    print("start train...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for user, pos, nega in train_loader:
            optimizer.zero_grad()
            
            user, pos, nega = user.to(args.device), pos.to(args.device), nega.to(args.device),
            bprloss = model(user, pos, nega)
            
            bprloss.backward()
            optimizer.step()
            total_loss += bprloss.item()
        
        model.eval()
        with torch.no_grad():
            score = torch.mm(model.user_final_embedding, model.item_final_embedding.T)
            score += dataset.train_mask.to(args.device)
            _, idx = torch.topk(score, k=10)
            pred_items = idx.to("cpu").detach().numpy()
            recall = recall_at_10(dataset.valid, pred_items)
            
            if recall > best_recall:
                best_recall = recall
                if not os.path.exists(args.save_dir_path):
                    os.makedirs(args.save_dir_path)
                torch.save(model, f"{args.save_dir_path}/best_model.pt")
                torch.save(model.state_dict(), f"{args.save_dir_path}/state_dict.pt")
        
        print(f'epoch: {epoch}, loss: {total_loss/len(train_loader)}, recall : {recall}')        

            
def inference(dataset, args):
    
    model = torch.load(f"{args.save_dir_path}/best_model.pt") 
    model.load_state_dict(torch.load(f"{args.save_dir_path}/state_dict.pt"))
    model.to(args.device)
    model.eval()
    topk = []  

    print("Start Inference...")
    with torch.no_grad():
        score = torch.mm(model.user_final_embedding, model.item_final_embedding.T)
        score += dataset.sub_mask.to(args.device)
        _, idx = torch.topk(score, k=10)
        pred_items = idx.to("cpu").detach().numpy()
        
        for index, items in enumerate(pred_items):
            for item in items:
                topk.append((dataset.user_map[index], dataset.item_map[item]))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pd.DataFrame(topk, columns=["user", "item"]).to_csv("lightgcn_submission.csv", index=False)
    
    print('Done.')          