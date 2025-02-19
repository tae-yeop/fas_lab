import logging
from datetime import datetime
from types import MethodType
import sys
import os
import yaml
import argparse

import torch
import numpy as np
import random

try:
    import wandb
    wandb_avail = True
except ImportError:
    wandb_avail = False
    # pass

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

def get(self, key, default=None):
    return getattr(self, key, default)

def get_args_with_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='path to config file')
    args = parser.parse_args()
    assert args.config is not None

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    args.get = get.__get__(args)
    
    return args


# from types import SimpleNamespace

# def dict_to_simplenamespace(d):
#     """
#     재귀적으로 딕셔너리를 SimpleNamespace로 변환하는 함수.
#     중첩된 딕셔너리에 대해서도 작동합니다.
#     """
#     if isinstance(d, dict):
#         for key, value in d.items():
#             d[key] = dict_to_simplenamespace(value)
#         return SimpleNamespace(**d)
#     elif isinstance(d, list):
#         return [dict_to_simplenamespace(item) for item in d]
#     else:
#         return d

# def get_args_with_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_rank', type=int)
#     parser.add_argument('--config', type=str, default='configs/train.yaml', help='path to config file')
#     args = parser.parse_args()
#     assert args.config is not None

#     with open(args.config, 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     config = dict_to_simplenamespace(config)
    
#     for key in vars(config):
#         for k, v in vars(getattr(config, key)).items():
#             setattr(args, k, v)
#     return args


def get_args_with_config_omegaconf():
    try:
        from omegaconf import OmegaConf
        args = OmegaConf.load('configs/train.yaml')
        OmegaConf.set_struct(args, True)
        return args
    except ImportError:
        return get_args_with_config()

def save_args_to_yaml(args, filename='saved_config.yaml'):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def log_eval(self, idx, loss, acc):
    self.info(f'{idx} iteration | loss : {loss} | acc : {acc}')
    
def get_logger(expname, log_path, file_log_mode='a'):
    logger = logging.getLogger('TEST')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)05s %(message)s \n\t--- %(filename)s line: %(lineno)d in %(funcName)s", '%Y-%m-%d %H:%M:%S')

    # 터미널 용 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 파일 저장용 핸들러
    file_handler = logging.FileHandler(f'{log_path}/{expname}.log', mode=file_log_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 시작 메시지
    start_message = f"\n\n{'=' * 50}\nSession Start: {expname} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 50}"
    logger.info(start_message)

    logger.log_eval = MethodType(log_eval, logger)

    return logger

def check_model_device(model):
    cpu_params = []
    cuda_params = []
    
    for name, param in model.named_parameters():
        if param.is_cuda:
            cuda_params.append(name)
        else:
            cpu_params.append(name)
    
    logger.info("CPU Parameters:")
    for param_name in cpu_params:
        logger.info(f"- {param_name}")
    
    logger.info("\nCUDA Parameters:")
    for param_name in cuda_params:
        logger.info(f"- {param_name}")
    
    if cpu_params and cuda_params:
        logger.info("\nModel parameters are located on both CPU and CUDA devices.")
    elif cpu_params:
        logger.info("\nAll model parameters are located on CPU.")
    else:
        logger.info("\nAll model parameters are located on CUDA.")