# 그냥 I2I train

import os
import numpy as np
from tqdm import tqdm
import warnings

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
from torchvision.utils import save_image

from training.general import set_random_seeds, set_torch_backends, get_args_with_config, save_args_to_yaml, get_logger
from training.optimizers import optimizer_dict, prepare_optimizer_params
from training.objectives import loss_dict #DenoiserLoss, NoiserLoss, multi_VGGPerceptualLoss, objective_dict
from dataset.datasets import (dataset_dict, MoirePairMixDataset, CustomDistributedSampler)
from dataset.transformations import get_transform
from models.models import model_dict


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    args = get_args_with_config()
    # ============================ Distributed Setting ============================
    dist.init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK']) # dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK']) # device_id = global_rank % torch.cuda.device_count()
    world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # print('local_rank', local_rank)
    # ============================ Basic Setting ==================================
    seed = args.random_seed
    if seed is not None:
        set_random_seeds(seed) 

    set_torch_backends(args)

    
    if global_rank == 0:
        log_path = f'./logs/{args.expname}'
        os.makedirs(log_path, exist_ok=True)
        save_args_to_yaml(args, f'{log_path}/config.yaml')

    # wandb setup
    if global_rank == 0 and args.wandb['use']:
        try:
            import wandb
            wandb.login(key=args.wandb['key'], host=args.wandb['host'])
            wandb.init(project="i2i", entity="tyk", group=args.expname)
            wandb.config.update()
        except ImportError:
            args.wandb['use'] = False

    # logger setup
    if global_rank == 0:
        logger = get_logger(args.expname, log_path)
        logger.info(f'{global_rank}, {local_rank}, {world_size}, {device}, {seed}')


    # ============================ Dataset =========================================
    assert args.train_batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
    trans = get_transform(trans_type='UHDM_train', train_img_size=args.train_img_size)
    dataset_list = []
    for dataset_name in args.train_dataset_list:
        dataset_list.append(dataset_dict[dataset_name](transformation=trans))
    trainset = MoirePairMixDataset(dataset_list)

    for dataset_name in args.test_dataset_list:
        if dataset_name == 'uhd':
            testset = dataset_dict[dataset_name]('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test', transformation=trans)
        else:
            continue

    train_sampler = CustomDistributedSampler(trainset, rank=global_rank, num_replicas=world_size, shuffle=True)
    test_sampler = CustomDistributedSampler(testset, rank=global_rank, num_replicas=world_size, shuffle=False, is_validation=True)


    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, sampler=train_sampler, 
                            num_workers=16, pin_memory=True, shuffle=False, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=args.val_batch_size, sampler=test_sampler, 
                             num_workers=16, pin_memory=True, shuffle=False, persistent_workers=True)

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model_class = model_dict[args.model]
    model = model_class(args, device)#.to(device)
    # print('1', model.denoiser is not None)
    # print(next(model.denoiser.parameters()).device)
    model.wrap_ddp(local_rank)
    # print('2', model.denoiser is not None)
    # -----------------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------------
    # model_module_dict = model.get_all_modules(depth=1)
    model_module_dict = model.get_module_in_ddp()
    
    # print('model_module_dict', model_module_dict.keys())
    # print('3', model.denoiser is not None)
    optimizer_class = optimizer_dict[args.optimizer]
    # params_to_optimize = []
    # print(model_module_dict.keys())
    # for model_name, instance in model_module_dict.items():
    #     params_to_optimize.append({"params": instance.parameters(),
    #                                 "lr" : args.learning_rates[model_name]})

    params_to_optimize = prepare_optimizer_params(model_module_dict, args.learning_rates)
    # if global_rank == 0:
    #     print(params_to_optimize)
    optimizer = optimizer_class(params_to_optimize, **args.optimizer_args)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.scheduler_gamma)

    # objective = objective_dict[args.objective['type']](args.objective, device)
    # model.set_objectives(objective)

    model.set_objectives(args.objective, device)
    # print('4', model.denoiser is not None)
    # denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=args.noiser_lr, weight_decay=5e-4)
    # denoiser_scheduler = StepLR(denoiser_optimizer, step_size=1, gamma=args.noiser_gamma)

    # noiser_optimizer = optim.Adam(noiser.parameters(), lr=args.denoiser_lr, weight_decay=5e-4)
    # noiser_scheduler = StepLR(noiser_optimizer, step_size=1, gamma=args.denoiser_gamma)
    # -----------------------------------------------------------------------------
    # Training setup
    # -----------------------------------------------------------------------------
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
    #                                             output_device=local_rank)

    total_idx = 0
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -----------------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------------
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    # print(model.denoiser is not None)
    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False) if global_rank == 0 else train_loader
        for idx, data in enumerate(train_loader):
            # print('idx', idx)
            logger.info(f'idx : {idx}') if global_rank == 0 else None
            total_idx += 1
            clean_img = data['clean_img'].to('cuda', non_blocking=True) # inputs.cuda()
            noisy_img = data['moire_img'].to('cuda', non_blocking=True) # labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            # denoiser_optimizer.zero_grad(set_to_none=True)
            # noiser_optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = model.train_step(clean_img, noisy_img, total_idx)
                # output = model(clean_img, noisy_img)
                # logger.info(f'loss : {output["loss"]}')
                loss = output['loss']
                # logger.info(f'loss : {loss}')
            if args.fp16:
                scaler.scale(loss).backward()
                # scaler.step(noiser_optimizer)
                # scaler.step(denoiser_optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                # noiser_optimizer.step()
                # denoiser_optimizer.step()
            if total_idx % args.checkpoint_steps == 0 and global_rank == 0:
                # print('total_idx, loss_n, loss_d', total_idx, loss_n, loss_d)
                logger.info(f'total_idx, loss : {total_idx}, {loss}')
                # model_module_dict = model.module.get_all_modules(depth=1)

                model_module_dict = model.get_module_in_ddp()
                dict_states = {
                    module_name : getattr(instance, 'state_dict')() for module_name, instance in model_module_dict.items() if hasattr(instance, 'state_dict')
                }
                dict_states2 = {
                    'optimizer' : optimizer.state_dict(),
                    'step' : total_idx,
                    'exp_name' : args.expname
                }
                dict_states.update(dict_states2)
                torch.save(dict_states, f'{log_path}/checkpoint-{total_idx}.ckpt')

            if epoch % args.validation_epoch == 0:
            # if total_idx % args.validation_steps == 0:
                # print('indices_list', indices_list)
                model.eval()
                ddp_loss = torch.zeros(2, device='cuda')
                test_loader = tqdm(test_loader, desc = f'Epoch {epoch}', leave=False) if global_rank == 0 else test_loader
                with torch.no_grad():
                    for test_data in test_loader:
                        clean_img = test_data['clean_img'].to('cuda', non_blocking=True) # inputs.cuda()
                        noisy_img = test_data['moire_img'].to('cuda', non_blocking=True) # labels.cuda()
                        # clean_hat1, _, _ = denoiser(noisy_img)
                        # noisy_hat1, _, _ = noiser(clean_img)
                        output = model.val_step(clean_img, noisy_img, total_idx)
                        clean_img = clean_img # output['clean']
                        noisy_img = output['noisy']
                        ddp_loss[0] += output['loss'].item()
                        ddp_loss[1] += len(test_data)
                        if global_rank == 0:
                            save_image(clean_img, f'{log_path}/clean_img_{total_idx}.png')
                            save_image(noisy_img, f'{log_path}/noise_img_{total_idx}.png')
                    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                    val_avg_loss = ddp_loss[0] / ddp_loss[1]
                    if global_rank == 0:
                        logger.info(f'val_avg_loss : {val_avg_loss}')
        # denoiser_scheduler.step()
        # noiser_scheduler.step()
        scheduler.step()

    dist.barrier()
    init_end_event.record()

    if global_rank==0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    # wandb logout
    if global_rank == 0 and args.wandb:
        wandb.finish()
    dist.destroy_process_group()
            

            
            
            