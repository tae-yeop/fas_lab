import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from config import TraningConfig
from datasets import get_dataloader
from models import Depth3DCNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    cfg = TraningConfig()

    dataloader = get_dataloader(cfg.json_path, cfg.batch_size)

    fabric = Fabric(
        accelerator='cuda', 
        num_nodes=args.nnodes,
        devices=args.ngpus,
        strategy=cfg.pl_strategy, 
        precision=cfg.pl_precision,
    )

    fabric.launch()
    fabric.seed_everything(cfg.seed)

    if cfg.model == '3dcnn':
        model = Depth3DCNN().cuda()
    else:
        pass

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dataloader:
            x, y = batch
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        
        print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")