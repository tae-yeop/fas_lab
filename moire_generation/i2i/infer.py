import argparse
import yaml
import os

import torch
from PIL import Image
from torchvision import transforms

from dataset import ToTensor
def get_parser():
    parser = argparse.ArgumentParser(description='IMAGE_DEMOIREING')
    parser.add_argument('--local_rank', default=1, type=int)
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml', help='path to config file')
    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


if __name__ == '__main__':
    args = get_parser()
    print(args)
    print(args.local_rank)
    print(args.inference)
    print(args.extractor_loss)

    moire = Image.open('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test/0106_moire.jpg')
    clean = Image.open('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test/0106_gt.jpg')
    moire, clean = ToTensor()((moire, clean))
    from torchvision.utils import save_image
    diff = moire - clean
    save_image(diff.clip(0, 1), 'diff.png')

    moire_hat = clean + diff
    print("Moiré noise value range:", diff.min().item(), "to", diff.max().item())

    moire_noise_abs = torch.abs(diff)

    # Moiré 패턴 노이즈를 PIL 이미지로 변환
    save_image(moire_noise_abs.clip(0,1), 'moire_hat2.png')

    save_image(moire_hat, 'moire_hat.png')
    save_image(moire, 'moire.png')

    clean2 = Image.open('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test/0146_gt.jpg')
    clean3 = Image.open('/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test/0170_gt.jpg')
    clean2, clean3 = ToTensor()((clean2, clean3))
    moire2 = clean2 + diff
    moire3 = clean3 + diff
    save_image(moire2, 'moire2.png')
    save_image(moire3, 'moire3.png')

    
    