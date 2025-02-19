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

# from training.general import set_random_seeds, set_torch_backends, get_args_with_config, save_args_to_yaml, get_logger
# from training.optimizers import optimizer_dict
# from training.objectives import DenoiserLoss, NoiserLoss, multi_VGGPerceptualLoss, objective_dict
# from dataset.datasets import (dataset_dict, MoirePairMixDataset, CustomDistributedSampler)
# from dataset.transformations import get_transform
# from models.models import model_dict

if __name__ == '__main__':
    print(torch.__version__.split('+')[0].split('.'))
    print(tuple(map(int, torch.__version__.split('+')[0].split('.'))))
    # args = get_args_with_config()
    # print(args.objective['type'])

    # for loss in args.denoiser_loss:
    #     print(loss)

#     from diffusers import DiffusionPipeline

#     pipeline = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, cache_dir= '/purestorage/project/tyk/3_CUProjects/FAS/i2i-translation/tmp'
# )
#     print(pipeline.scheduler.alphas_cumprod)