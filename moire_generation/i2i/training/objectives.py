import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os

from focal_frequency_loss import FocalFrequencyLoss as FFL

# -----------------------------------------------------------------------------
# I2I Loss
# -----------------------------------------------------------------------------
class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)
        
        return loss1+loss2+loss3     

class single_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(single_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, gt1, feature_layers=[2]):
        # gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        # gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        # loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        # loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)
        
        return loss1

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class PerPixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perpixelloss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_moire, moire):
        return self.perpixelloss(pred_moire, moire)


def get_kernel(kernel_size, index):
    if index < kernel_size:
        start = (index,0)
        end = (kernel_size - 1 - index, 6)
    else:
        start = (6, index - kernel_size + 1)
        end = (0, kernel_size * 2 - index - 2)
    
    filter = np.zeros((kernel_size,kernel_size))
    
    filter = cv2.line(filter,start, end,1,1)
    filter = filter/np.sum(filter)
    return filter

def generate_filters(kernel_size=7, num_kernels=14):
    total_diff = (kernel_size - 1)*2
    kernels = []
    for i in range(total_diff):
        kernels.append(get_kernel(kernel_size, i))
    
    if num_kernels - total_diff > 0:
        for i in range(num_kernels - total_diff):
            kernels.append(kernels[0])
    
    return kernels

class DirectionalLoss(nn.Module):
    def __init__(self, kernel_size=7, num_kernels=14):
        super().__init__()
        filters = generate_filters(kernel_size=kernel_size, num_kernels=num_kernels)
        conv_weighs = [torch.FloatTensor(v).unsqueeze(0).unsqueeze(0) for v in filters]
        self.conv_weights = [nn.Parameter(data=v, requires_grad=False) for v in conv_weighs]
        self.kernel_size=kernel_size

        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, pred_moire, moire):
        loss_list = []
        for conv_weight in self.conv_weights:
            conv_weight = conv_weight.to(pred_moire.device)
            conv_pre = F.conv2d(pred_moire, conv_weight, padding=(self.kernel_size-1)//2)
            conv_gt = F.conv2d(moire, conv_weight, padding=(self.kernel_size-1)//2)

            temp_loss = self.l1_loss(conv_pre,conv_gt)

            loss_list.append(torch.mean(temp_loss))

        return sum(loss_list)/len(loss_list)

class DistributionLoss(nn.Module):
    def __init__(self, kernel_size=7,stride=1):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size),padding=(kernel_size-1)//2,stride=stride)
        self.l1_loss = torch.nn.SmoothL1Loss()

    def forward(self, pred_moire, moire):
        eps = 0.00000001
        pred_moire = self.unfold(pred_moire)
        moire = self.unfold(moire)
        
        pred_moire = pred_moire - torch.mean(pred_moire,dim=1,keepdim = True)
        pred_moire = (torch.mean(pred_moire**2,dim=1)+eps)**0.5

        moire = moire - torch.mean(moire,dim=1,keepdim = True)
        moire = (torch.mean(moire**2,dim=1)+eps)**0.5

        loss = torch.mean(self.l1_loss(pred_moire,moire))
        return loss


def proxy_loss(clean_hat1, moire_hat, moire_img):
    return clean_hat1 + moire_hat - moire_img


class CycleLoss(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.loss_dict = {"smooth_l1_loss": PerPixelLoss().to(device),
                         "proxy_loss":proxy_loss,
                         "direction_loss":DirectionalLoss().to(device),
                         "dist_loss":DistributionLoss().to(device),
                         "perceptual_loss": multi_VGGPerceptualLoss(params['lam'], params['lam_p']).to(device)}


    def forward(self, pred_hat1, pred_hat2, pred_hat3, target_img, cyc_pred_hat1, cyc_pred_hat2, cyc_pred_hat3):
        total_loss = 0
        total_loss += self.loss_dict["perceptual_loss"](pred_hat1, pred_hat2, pred_hat3, target_img)
        total_loss += self.loss_dict["perceptual_loss"](cyc_pred_hat1, cyc_pred_hat2, cyc_pred_hat3, target_img)
        return total_loss


class PatchFreqLoss(nn.Module):
    def __init__(self, patch_size, norm):
        super().__init__()
        self.patch_size = patch_size
        self.norm = norm

    def image_to_patches(self, image):
        c, h, w = image.size()
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(c, -1, patch_size, patch_size)
        return patches.permute(1, 0, 2, 3)

    def forward(self, pred, gt):
        orig_patches = orig_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        pred_patches = pred_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)


class PatchFreqLoss(nn.Module):
    def __init__(self, patch_size, norm):
        super().__init__()
        self.patch_size = patch_size
        self.norm = norm

    def image_to_patches(self, img):
        batch_size, c, _, _ = img.size()
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (batch_size, num_patches, C, patch_size, patch_size)
        patches = patches.reshape(-1, c, self.patch_size, self.patch_size)  # (batch_size * num_patches, C, patch_size, patch_size)
        return patches

    def forward(self, pred, target):
        pred_patches = self.image_to_patches(pred)
        target_patches = self.image_to_patches(target)
        
        pred_fft = torch.fft.rfft2(pred_patches)
        target_fft = torch.fft.rfft2(target_patches)
        if self.norm == 'l1':
            freq_diffs = torch.norm(pred_fft - target_fft, p=1, dim=(-3, -2, -1))
        elif self.norm == 'l2':
            freq_diffs = torch.norm(pred_fft - target_fft, p=2, dim=(-3, -2, -1))
        else:
            raise ValueError("norm type error")
            # loss = torch.mean(torch.abs(pred_fft - target_fft))
        loss = torch.mean(freq_diffs)
        return loss


# -----------------------------------------------------------------------------
# GAN Loss
# -----------------------------------------------------------------------------
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_hinge(d_logit_real, d_logit_fake):
    return -torch.mean(d_logit_fake)

def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def d_r1_loss(real_logit, real_img, r1_gamma):
    grad_real, = grad(outputs=real_logit.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return 0.5 * r1_gamma * grad_penalty


# -----------------------------------------------------------------------------
# loss dict
# -----------------------------------------------------------------------------
loss_dict = {'single_vgg_loss': single_VGGPerceptualLoss, 'multi_vgg_loss' : multi_VGGPerceptualLoss, 'ffl': FFL, 'patch_freq' : PatchFreqLoss, 'cyc_loss' : multi_VGGPerceptualLoss}
gan_loss_dict = {'hinge': [g_hinge, d_hinge], 'ns': [g_nonsaturating_loss, d_logistic_loss], 'r1': d_r1_loss}
# class Objective():
#     def __init__(
#         self,
        
#     ):
#         super().__init__()

#     def calc_d_loss(self, ):
#         if exists(self.cyc_loss): d_loss += self.


#     def cyc_loss():
        
# class ExtractorLoss(nn.Module):
#     def __init__(self, loss_dict):
#         super().__init__()
#         self.loss_args = loss_args
#         self.loss_dict = {"smooth_l1_loss": PerPixelLoss(),
#                          "proxy_loss":proxy_loss,
#                          "direction_loss":DirectionalLoss(),
#                          "dist_loss":DistributionLoss(),
#                          }

#     def forward(self, clean_img, noisy_img, noise_hat, clean_hat):
#         total_loss = 0
#         pseudo_noise_img = noisy_img - 
#         for loss_name, loss_info in self.loss_args.items():
#             loss_fn = self.loss_functions[loss_name]
#             loss_weight = loss_info["lambda"]
            

if __name__ == '__main__':
    ...
    # Loss test 해보기
    # gt vs moire간의 loss test
    # /purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train/pair_01/0000_gt.jpg
    # 




# class DenoiserLoss(nn.Module):
#     def __init__(self, loss_args, device):
#        super().__init__()
#        self.loss_args = loss_args
#        vgg_loss_dict = self.loss_args['multi_vgg_loss']
#        # dict에 담으면 장치에 제대로 가지 않음
#        self.loss_dict = {"smooth_l1_loss": PerPixelLoss().to(device),
#                          "proxy_loss":proxy_loss,
#                          "direction_loss":DirectionalLoss().to(device),
#                          "dist_loss":DistributionLoss().to(device),
#                          "perceptual_loss": multi_VGGPerceptualLoss(vgg_loss_dict['l1_lambda'], vgg_loss_dict['vgg_lambda']).to(device)}
#     def forward(self, clean_hat1, clean_hat2, clean_hat3, clean_img, clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3):
#         total_loss = 0
#         total_loss += self.loss_dict["perceptual_loss"](clean_hat1, clean_hat2, clean_hat3, clean_img)
#         total_loss += self.loss_dict["perceptual_loss"](clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3, clean_img)
#         return total_loss

# class NoiserLoss(nn.Module):
#     def __init__(self, loss_args, device):
#         super().__init__()
#         self.loss_args = loss_args
#         vgg_loss_dict = self.loss_args['multi_vgg_loss']
#         self.loss_dict = {"smooth_l1_loss": PerPixelLoss().to(device),
#                          "proxy_loss":proxy_loss,
#                          "direction_loss":DirectionalLoss().to(device),
#                          "dist_loss":DistributionLoss().to(device),
#                          "perceptual_loss": multi_VGGPerceptualLoss(vgg_loss_dict['l1_lambda'], vgg_loss_dict['vgg_lambda']).to(device)}


#     def forward(self, noise_hat1, noise_hat2, noise_hat3, noise_img, noise_cyc_hat1, noise_cyc_hat2, noise_cyc_hat3):
#         total_loss = 0
#         total_loss += self.loss_dict["perceptual_loss"](noise_hat1, noise_hat2, noise_hat3, noise_img)
#         total_loss += self.loss_dict["perceptual_loss"](noise_cyc_hat1, noise_cyc_hat2, noise_cyc_hat3, noise_img)
#         return total_loss

# class Loss1():
#     def __init__(self, objective_args, device):
#         denoiser_dict = objective_args['denoiser_loss']
#         noiser_dict = objective_args['noiser_loss']
#         self.denoiser_loss = loss_dict[denoiser_dict['type']](**denoiser_dict['params']).to(device)
#         self.noiser_loss = loss_dict[noiser_dict['type']](**noiser_dict['params']).to(device)


#     def __call__(self, model_output):
#         total_loss = 0
#         denoiser_loss = 0
#         denoiser_loss += self.denoiser_loss(model_output['noise_hat1'], model_output['noise_hat2'], 
#                                             model_output['noise_hat3'], model_output['noise_img'])
#         denoiser_loss += self.denoiser_loss()
        
#         total_loss += ()
#         return total_loss