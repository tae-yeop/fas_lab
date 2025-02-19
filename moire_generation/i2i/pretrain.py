import os
import numpy as np
import random 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda import amp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms

from dataset import _list_image_files_recursively, UHDMDataset, LCDMoireDataset, FHDMIDataset, MoireDetDataset, VDDDataset, MoirePairMixDataset
from loss import proxy_loss, PerPixelLoss, DirectionalLoss, DistributionLoss
import argparse
import yaml

from denoiser import Denoiser
from extractor import MoireDetExtractor

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def set_torch_backends(args):
    # Ampare architecture 30xx, a100, h100,..
    if torch.cuda.get_device_capability(0) >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    if args.inference : torch.set_grad_enabled(False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml', help='path to config file')
    args = parser.parse_args()
    assert args.config is not None

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    return args

if __name__ == '__main__':
    args = get_args()
    
    # ============================ Distributed Setting ============================
    dist.init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK']) # dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK']) # device_id = global_rank % torch.cuda.device_count()
    world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    print(global_rank, local_rank, world_size, device)

    # ============================ Basic Setting ==================================
    seed = args.random_seed
    if seed is not None:
        set_random_seeds(seed) 

    set_torch_backends(args)


    # ============================ wandb setup ==================================
    # wandb.login(key=wandb_info['key'], host=wandb_info['host'])

    
    # ============================ Dataset =========================================
    assert args.batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
    trans = get_transform(trans_type='UHDM_train')
    
    ls = _list_image_files_recursively('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train')
    uhdm = UHDMDataset(ls, transformation=trans)
    lcm = LCDMoireDataset(transformation=trans)
    fh = FHDMIDataset(transformation=trans)
    md = MoireDetDataset(transformation=trans)
    vdd = VDDDataset(transformation=trans)
    trainset = MoirePairMixDataset([uhdm, lcm, fh, md, vdd])

    
    ls = _list_image_files_recursively('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test')
    testset = UHDMDataset(ls, transformation=trans)


    train_sampler = DistributedSampler(trainset, rank=global_rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(testset, rank=global_rank, num_replicas=world_size, shuffle=False)


    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, 
                            num_workers=8, pin_memory=True, shuffle=False, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, 
                             num_workers=8, pin_memory=True, shuffle=False, persistent_workers=True)
    # ============================ Models ============================================
    denoiser = Denoiser().to(device)
    extractor = MoireDetExtractor().to(device)

    denoiser = nn.parallel.DistributedDataParallel(denoiser, device_ids=[local_rank], 
                                                output_device=local_rank)

    extractor = nn.parallel.DistributedDataParallel(extractor, device_ids=[local_rank], 
                                                output_device=local_rank)


    # ============================ Traning setup ======================================
    # loss for denoiser
    # l1_loss = nn.L1Loss()
    # mse_loss = nn.MSELoss()
    perceptual_loss = multi_VGGPerceptualLoss(args.multi_loss_l1, args.multi_loss_p)
    # loss for extractor
    

    denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=0.001, weight_decay=5e-4)
    denoiser_scheduler = StepLR(denoiser_optimizer, step_size=1, gamma=args.gamma)


    extractor_optimizer = optim.Adam(extractor.parameters(), lr=0.001, weight_decay=5e-4)
    extractor_scheduler = StepLR(extractor_optimizer, step_size=1, gamma=args.gamma)


    total_idx = 0
    total_loss = self.perpixelweight * perpixel_loss + self.directionalweight * directional_loss + self.distributionweight * distribution_loss
    # ============================ Train ==============================================
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(args.num_epochs):
        denoiser.train()
        extractor.train()
        train_sampler.set_epoch(epoch)

        train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False) if rank == 0 else train_loader
        for idx, data in enumerate(train_loader):
            total_idx += 1
            denoiser_loss = 0
            extraction_loss = 0
            clean_img = data['clean_img'].to('cuda', non_blocking=True) # inputs.cuda()
            noisy_img = data['moire_img'].to('cuda', non_blocking=True) # labels.cuda()
            extractor_optimizer.zero_grad(set_to_none=True)
            denoiser_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                clean_hat1, clean_hat2, clean_hat3 = denoiser(noisy_img)
                noise_hat = extractor(noisy_img)

                denoiser_loss += perceptual_loss(clean_hat1, clean_hat2, clean_hat3, clean_img)
                extraction_loss += proxy_loss(clean_hat1, noise_hat, noisy_img)
            if scaler:
                denoiser_loss = scaler.scale(denoiser_loss)
                denoiser_loss.backward()
                scaler.step(extractor_optimizer)
                scaler.update()

                extraction_loss = scaler.scale(extraction_loss)
                extraction_loss.backward()
                
            else:
                denoiser_loss.backward()
                denoiser_optimizer.step()

                extraction_loss.backward()
                extractor_optimizer.step()

            if total_idx % args.validation_steps == 0:
                denoiser.eval()
                extractor.eval()
                test_loader = tqdm(test_loader, desc = f'Epoch {epoch}', leave=False) if rank == 0 else test_loader
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        clean_img = data['clean_img'].to('cuda', non_blocking=True) # inputs.cuda()
                        noisy_img = data['moire_img'].to('cuda', non_blocking=True) # labels.cuda()



        scheduler.step()
                    
    dist.barrier()
    init_end_event.record()

    if global_rank==0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    dist.destroy_process_group()

    
    
    
    