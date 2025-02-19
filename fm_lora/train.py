import argparse
import wandb
import os
import time

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from backbone import get_model, get_output_dim

from dataclasses import dataclass
@dataclass
class TrainingConfig:
    model_name: str = "clip"
    use_lora: bool = True
    train_scratch: bool = False
    num_epoch:int = 40
    lr_model:float = 1e-6
    lr_header:float = 1e-3
    momentum:float = 0.9
    weight_decay:float = 0.05
    eta_min:float = 0 
    lora_dropout:float =0.4
    lora_r:int =8
    lora_a:int =8
    max_norm:float =5
    loss:str ="BinaryCrossEntropy"
    global_step:int =0
    scheduler_type:str ="cosine"
    warmup:bool =True
    num_warmup_epochs:int =5
    T_0:int =5
    T_mult:int =2
    model_name:str = "clip"
    lr_func_drop:list = [22, 30, 40]
    batch_size:int =64
    lora_bias:str ="none"
    lora_target_modules:list =["q", "v"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--ngpus", type=int)
    parser.add_argument("--resume_ckpt_path", type=str)
    args = parser.parse_args()

    cfg = TrainingConfig()

    fabric = Fabric(
        accelerator='cuda', 
        num_nodes=args.nnodes,
        devices=args.ngpus, 
        strategy=config.pl_strategy,
        precision=config.pl_precision,
    )


    train_loader, test_loader = get_dataloader()
    train_loader = 

    model = get_model(cfg, fabric.device)

    fabric.print(sum([p.nueml() for p in model.parameters() if p.requires_grad]))
    if cfg.use_lora:
        apply_lora_model(local_rank, model, **cfg)
    elif cfg.train_scratch:
        model.backbone.initialize_parameters()
        fabric.print("Model initialized from scratch")
    if cfg.use_lora or cfg.train_scratch:
        model.train()

    # Header
    output_dim = get_output_dim(**cfg)
    header = get_header()
    header.train()

    if cfg.training_type == "PAD_training" or cfg.training_type == "PAD_training_scratch":
        traintype = "PAD_training"
    elif cfg.training_type == "PAD_training_only_header":
        traintype = "PAD_training_only_header"
    else:
        ValueError()


    