import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
import os

from config import TraningConfig
from datasets import get_dataloader
from models import SimpleModel, ViT_Model


def init_optimizer(args):
    optimizer_class = None
    if 'offload' in args.pl_strategy:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer_class = DeepSpeedCPUAdam
    else:
        optimizer_class = optim.AdamW


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    cfg = TraningConfig()

    dataloader = get_dataloader(cfg.json_path, cfg.batch_size)

    # wandb.login(key=cfg.wandb_key, host=cfg.wandb_host)
    # wandb_logger = WandbLogger(
    #     project=cfg.wandb_project_name, 
    #     name=cfg.wandb_run_name + '-' + os.environ.get('SLURM_JOBID', ''), 
    #     config=cfg)

    fabric = Fabric(
        accelerator='cuda', 
        num_nodes=args.nnodes,
        devices=args.ngpus,
        strategy=cfg.pl_strategy, 
        precision=cfg.pl_precision,
    )

    fabric.launch()
    fabric.seed_everything(cfg.seed)

    if cfg.model == 'simple':
        model = SimpleModel().cuda()
    elif cfg.model == 'vit':
        model = ViT_Model().cuda()
    elif cfg.model == 'gnn':
        model = LandmarkGNN(in_channels=2, hidden_dim=32, out_dim=1).cuda()
    else:
        pass

    
    criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss() for gnn?
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        for batch in dataloader:
            if cfg.model == 'gnn':
                input_data = batch
                label = batch.label.view(-1, 1) 
            else:
                input_data, label = batch
                input_data = input_data.float().cuda()
                label = label.long().cuda()

            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
