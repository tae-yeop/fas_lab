import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from PIL import Image, ImageFile
from PIL.ImageOps import exif_transpose
import os
import glob
import random

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def pil_rgb_convert(image):
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    return image

def _list_image_files_recursively(data_dir):
    file_list = []
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('gt.jpg'):
                file_list.append(os.path.join(home, filename))
    file_list.sort()
    return file_list
    
class UHDMDataset(Dataset):
    def __init__(self, root='/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train', transformation=None):
        super().__init__()
        # self.gt_file_list = gt_file_list
        self.gt_file_list = _list_image_files_recursively(root)
        self.transformation = transformation
        
    def __len__(self):
        return len(self.gt_file_list)

    def __getitem__(self, idx):
        # 손상된 이미지 처리
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}

        # 클린 이미지
        path_tar = self.gt_file_list[idx]
        number = os.path.split(path_tar)[-1][0:4]
        # 모이레 이미지 얻기
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'

        clean_img = pil_rgb_convert(exif_transpose(Image.open(path_tar)))
        moire_img = pil_rgb_convert(exif_transpose(Image.open(path_src)))
        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        # data['clean_img_path'] = path_src
        return data


class LCDMoireDataset(Dataset):
    def __init__(self, root='/purestorage/datasets/fas_moire_datasets/lcdmoire/val/', transformation=None):
        super().__init__()
        self.root = root
        clean_path_root = os.path.join(self.root, 'clean')
        self.clean_img_list = [os.path.join(clean_path_root, img) for img in os.listdir(clean_path_root) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 중간에 빠진게 있을 수 있어서 clean_img에 짝을 맞추도록 함
        # sprint(self.clean_img_list)

        self.moire_img_list = [clean_img_path.replace("clean", "moire") for clean_img_path in self.clean_img_list]
        # print(self.moire_img_list)
        self.transformation = transformation
        
    def __len__(self):
        return len(self.clean_img_list)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_clean = self.clean_img_list[idx]
        path_moire = self.moire_img_list[idx]

        clean_img = pil_rgb_convert(exif_transpose(Image.open(path_clean)))
        moire_img = pil_rgb_convert(exif_transpose(Image.open(path_moire)))

        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data

class FHDMIDataset(Dataset):
    def __init__(self, root='/purestorage/datasets/fas_moire_datasets/fhdmi', transformation=None, include_test=True):
        super().__init__()
        self.root = root
        self.train_root = os.path.join(self.root, 'train')

        clean_pattern = os.path.join(self.train_root, "target", "*.[pj][pn]g")
        self.clean_img_list = glob.glob(clean_pattern, recursive=True)

        # print(self.clean_img_list[:10])
        self.moire_img_list = [clean_img_path.replace('target', 'source').replace('tar', 'src') for clean_img_path in self.clean_img_list]

        # print(self.moire_img_list[:10])
        # self.clean_img = [os.path.join(clean_path_root, img) for img in os.listdir(os.path.join(self.train_root, 'target')) if img.lower().endswith(('.png', '.jpg', '.jpeg'))] 

        if include_test:
            self.test_root = os.path.join(self.root, 'test')
            clean_pattern = os.path.join(self.test_root, "target", "*.[pj][pn]g")
            test_clean_img_list = glob.glob(clean_pattern, recursive=True)
            tset_moire_img_list = [clean_img_path.replace('target', 'source').replace('tar', 'src') for clean_img_path in test_clean_img_list]
            self.clean_img_list += test_clean_img_list
            self.moire_img_list += tset_moire_img_list

            # print(self.clean_img_list[-10:])
            # print(self.moire_img_list[-10:])

        self.transformation = transformation
            
    def __len__(self):
        return len(self.clean_img_list)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_clean = self.clean_img_list[idx]
        path_moire = self.moire_img_list[idx]

        clean_img = pil_rgb_convert(exif_transpose(Image.open(path_clean)))
        moire_img = pil_rgb_convert(exif_transpose(Image.open(path_moire)))

        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data

    
class MoireDetDataset(Dataset):
    def __init__(self, root='/purestorage/datasets/fas_moire_datasets/moiredet', transformation=None):
        super().__init__()
        self.root = root
        
        H, W = 1944, 3264 
        white_img_tensor = torch.ones((3, H, W))
        self.white_img_pil = TF.to_pil_image(white_img_tensor)
        # save_image(white_img_tensor, 'img7.png')
        pattern = os.path.join(self.root, "layers_*", "*.[pj][pn]g")
        self.moire_img_list = glob.glob(pattern, recursive=True)

        self.transformation = transformation
        
    def __len__(self):
        # 18147개
        return len(self.moire_img_list)
    
    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}

        path_moire = self.moire_img_list[idx]

        clean_img = pil_rgb_convert(self.white_img_pil)
        moire_img = pil_rgb_convert(exif_transpose(Image.open(path_moire)))

        clean_img, moire_img = self.transformation((clean_img, moire_img))
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data

class VDDDataset(Dataset):
    def __init__(self, root='/purestorage/datasets/fas_moire_datasets/vdd', transformation=None, include_test=True):
        super().__init__()
        self.root = root
        train_clean_pattern = os.path.join(self.root, "*", "train", "target", "*.[pj][pn]g")
        self.clean_img_list = glob.glob(train_clean_pattern, recursive=True)

        # print(self.clean_img_list[:5], len(self.clean_img_list))
        self.moire_img_list = [clean_img_path.replace('target', 'source') for clean_img_path in self.clean_img_list]

        # print(self.moire_img_list[:5])

        if include_test:
            test_clean_pattern = os.path.join(self.root, "*", "test", "target", "*.[pj][pn]g")
            test_clean_img_list = glob.glob(test_clean_pattern, recursive=True)
            tset_moire_img_list = [clean_img_path.replace('target', 'source') for clean_img_path in test_clean_img_list]
            self.clean_img_list += test_clean_img_list
            self.moire_img_list += tset_moire_img_list


        # print(self.clean_img_list[:5], len(self.clean_img_list))
        # print(self.moire_img_list[:5])

        self.transformation = transformation
    def __len__(self):
        # train : 29638 , train +test : 34798
        return len(self.clean_img_list)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_clean = self.clean_img_list[idx]
        path_moire = self.moire_img_list[idx]

        clean_img = pil_rgb_convert(exif_transpose(Image.open(path_clean)))
        moire_img = pil_rgb_convert(exif_transpose(Image.open(path_moire)))

        clean_img, moire_img = self.transformation((clean_img, moire_img))
        
        data['clean_img'] = clean_img
        data['moire_img'] = moire_img
        return data

class MoirePairMixDataset(Dataset):
    def __init__(self, datasets, generator=None):
        """
        datasets: 데이터셋 인스턴스의 리스트
        """
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        self.generator = generator or torch.Generator()

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # idx를 사용하여 적절한 데이터셋과 그 내의 인덱스를 결정
        for i, dataset_length in enumerate(self.lengths):
            if idx < dataset_length:
                return self.datasets[i][idx]
            idx -= dataset_length
        raise IndexError("MixDataset: Index out of bounds")

class WeightedMoirePairMixDataset(Dataset):
    def __init__(self, datasets, probabilities):
        assert len(datasets) == len(probabilities), "Datasets and probabilities must have the same length"
        assert sum(probabilities) == 1, "Probabilities must sum to 1"
        self.datasets = datasets
        self.probabilities = probabilities
        self.cumulative_sizes = torch.tensor(probabilities).cumsum(0).tolist()
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Select a dataset based on the probabilities
        dataset_idx = next(i for i, cum_prob in enumerate(self.cumulative_sizes) if cum_prob > random.random())
        
        # Handle the case where datasets have different lengths
        idx = idx % len(self.datasets[dataset_idx])
        
        # Fetch the item from the selected dataset
        return self.datasets[dataset_idx][idx]
    
    
dataset_dict = {'uhd' :  UHDMDataset, 'lcm' : LCDMoireDataset, 'fhd' : FHDMIDataset, 'md' : MoireDetDataset, 'vdd': VDDDataset}


class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, is_validation=False):
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Distributed package is not available. Please setup distributed environment properly.")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Distributed package is not available. Please setup distributed environment properly.")
        #     rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.is_validation = is_validation
        self.epoch = 0

        if self.is_validation:
            self.divided = int(len(self.dataset) // self.num_replicas)
            remainder = len(self.dataset) % self.num_replicas
            self.num_samples = self.divided + remainder if self.rank == (self.num_replicas -1) else self.divided
            self.total_size = len(self.dataset)
        else:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.is_validation:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        if self.is_validation:
            if self.rank == (self.num_replicas-1):
                indices = indices[self.rank * self.divided : self.total_size]
            else:
                indices = indices[self.rank * self.divided : (self.rank + 1) * self.divided]
        else:  # 학습시
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
            
            
class CustomWeightedDistributedSampler():
    def __init__(self):
        ...
    def __iter__(self):
        ...

from torchvision.utils import save_image
if __name__ == '__main__':
    
    # data_dir = '/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train'
    # ls = _list_image_files_recursively(data_dir)
    trans = get_transform(trans_type='UHDM_train')
    # # trans = AlbumentationsTransform()
    # uhdm = UHDMDataset(ls, trans)
    # data = uhdm[0]
    # save_image(data['clean_img'], 'img1.png')
    # save_image(data['moire_img'], 'img2.png')


    # lcm = LCDMoireDataset(transformation=trans)
    # data = lcm[0]
    # save_image(data['clean_img'], 'img3.png')
    # save_image(data['moire_img'], 'img4.png')

    
    # fh = FHDMIDataset(transformation=trans)
    # data = fh[0]
    # save_image(data['clean_img'], 'img5.png')
    # save_image(data['moire_img'], 'img6.png')


    # md = MoireDetDataset(transformation=trans)
    # data = md[0]
    # save_image(data['clean_img'], 'img7.png')
    # save_image(data['moire_img'], 'img8.png')

    # vdd = VDDDataset(transformation=trans)
    # data = vdd[0]
    # save_image(data['clean_img'], 'img9.png')
    # save_image(data['moire_img'], 'img10.png')


    # mix = MoirePairMixDataset([uhdm, lcm, fh, md, vdd])
    # data = mix[0]
    # save_image(data['clean_img'], 'img11.png')
    # save_image(data['moire_img'], 'img12.png')
    data_dir = '/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test'
    ls = _list_image_files_recursively(data_dir)
    print(ls[:10])
    testset = UHDMDataset(data_dir, trans)
    test_sampler = CustomDistributedSampler(testset, rank=0, num_replicas=1, shuffle=False, is_validation=True)
    test_loader = DataLoader(testset, batch_size=4, sampler=test_sampler, 
                             num_workers=16, pin_memory=True, shuffle=False, persistent_workers=True)
    from tqdm import tqdm
    test_loader = tqdm(test_loader, desc = f'Epoch', leave=False)
    with torch.no_grad():
        for inputs, labels in test_loader:
            print(inputs, labels)
            break

    # params = {'batch_size': 64, 'shuffle':True, 'num_workers':8, 'pin_memory' : True,  'persistent_workers' : True}
    # dl = DataLoader(mix, **params)
    # # dl = DataLoader(dataset=mix, local_rank=0, **params)
    # from tqdm import tqdm  
    # # 배치 사이즈
    # print(len(dl))
    # for idx, data in tqdm(enumerate(dl), total=len(dl)):
    #     print(idx)
    #     continue

    # # 그냥 DL + params = {'batch_size': 64, 'shuffle':True, 'num_workers':8, 'pin_memory' : True,  'persistent_workers' : True} => 13분, 끝이 안나느데?
    # # 그냥 DLX + params = {'batch_size': 64, 'shuffle':True, 'num_workers':8, 'pin_memory' : True,  'persistent_workers' : True} => 15분
    # # 