"""
https://github.com/kobeshegu/ECCV2022_WaveGAN
"""
import random

import numpy as np
from torch import autograd
from torch import nn
from .blocks import *

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def batched_scatter(input, dim, index, src):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input, dim, index, src)


def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L 
    # array([[0.5, 0.5],
    #        [0.5, 0.5]])
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    # array([[-0.5,  0.5],
    #        [-0.5,  0.5]])
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    # array([[-0.5, -0.5],
    #        [ 0.5,  0.5]])
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    # array([[ 0.5, -0.5],
    #        [-0.5,  0.5]])
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)
    
    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

    
class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class LoFGAN(nn.Module):
    def __init__(self, config):
        super(LoFGAN, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])


class Discriminator(nn.Module):
    def __init__(self, config):
        pass

class Generator(nn.Module):
    def __init__(self, rate):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.fusion = LocalFusionModule(inplanes=128, rate=rate)

    def forward(self, xs):
        # b, k, C, H, W = xs.size()
        # xs = xs.view(-1, C, H, W)

        querys, skips = self.encoder(xs)
        # c, h, w = querys.size()[-3:]

        # querys = querys.view(b, k, c, h, w)

        # similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        # similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        # similarity = similarity_total / similarity_sum  # b*k

        # base_index = random.choice(range(k))
        # base_feat = querys[:, base_index, :, :, :]
        # feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)

        # fake_x = self.decoder(feat_gen, skips)
        print('querys.shape', querys.shape) # [64, 128, 16, 16])
        fake_x = self.decoder(querys, skips)
        return fake_x# , similarity, indices_feat, indices_ref, base_index

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

        self.pool1 = WavePool(32).cuda()
        self.conv2 = Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool2 = WavePool(64).cuda()
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool3 = WavePool2(128).cuda()
        self.conv4 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool4 = WavePool2(128).cuda()
        self.conv5 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

    def forward(self, x):
        #(24,3,128,128)
        skips = {}
        x = self.conv1(x)
        #(24,32,128,128)
        skips['conv1_1'] = x
        LL1, LH1, HL1, HH1 = self.pool1(x)
        # (24,64,64,64)
        skips['pool1'] = [LH1, HL1, HH1]
        x = self.conv2(x)
        #(24,64,64,64)
        # p2 = self.pool2(x)
        #（24,128,32,32）
        skips['conv2_1'] = x
        LL2, LH2, HL2, HH2 = self.pool2(x)
        #（24,128,32,32）
        skips['pool2'] = [LH2, HL2, HH2]

        x = self.conv3(x+LL1)
        #(24,128,32,32)
        # p3 = self.pool3(x)
        skips['conv3_1'] = x
        LL3, LH3, HL3, HH3 = self.pool3(x)
        #(24,128,16,16)
        skips['pool3'] = [LH3, HL3, HH3]
        #(24,128,32,32)
        x = self.conv4(x+LL2)
        #(24,128,16,16)
        skips['conv4_1'] = x
        LL4, LH4, HL4, HH4 = self.pool4(x)
        skips['pool4'] = [LH4, HL4, HH4]
        #(24,128,8,8)
        x = self.conv5(x+LL3)
        #(24,128,8,8)
        return x, skips

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block1 = WaveUnpool(128,"sum").cuda()
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block2 = WaveUnpool(128, "sum").cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block3 = WaveUnpool(64, "sum").cuda()
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block4 = WaveUnpool(32, "sum").cuda()
        self.Conv5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')

    def forward(self, x, skips):
        x1 = self.Upsample(x)
        x2 = self.Conv1(x1) # [64, 128, 16*2, 16*2])
        LH1, HL1, HH1 = skips['pool4']

        print("LH1, HL1, HH1", LH1.shape, HL1.shape, HH1.shape) # all : ([64, 128, 16, 16]) 
        
        LH2, HL2, HH2 = skips['pool3']
        print("LH2, HL2, HH2", LH2.shape, HL2.shape, HH2.shape) # all : ([64, 128, 32, 32])  

        LH3, HL3, HH3 = skips['pool2']
        print("LH3, HL3, HH3", LH3.shape, HL3.shape, HH3.shape) # all : ([64, 128, 64, 64])  
        
        # c, h, w = LH1.size()[-3:]
        b, c, h, w = LH1.size()
        k = 1
        LH1, HL1, HH1 = LH1.view(b, k,c, h, w).mean(dim=1), HL1.view(b, k,c, h, w).mean(dim=1), HH1.view(b, k,c, h, w).mean(dim=1)
        original1 = skips['conv4_1']
        original2 = skips['conv3_1']
        print("original1", original1.shape) # ([64, 128, 32, 32])
        print("original2", original2.shape) #([64, 128, 64, 64]) 

        print("x", x.shape) # ([64, 128, 16, 16])  
        x_deconv = self.recon_block1(x, LH1, HL1, HH1, original1)
        x2 = x_deconv + x2

        x3 = self.Upsample(x2)
        x4 = self.Conv2(x3)
        LH2, HL2, HH2 = skips['pool3']
        original2 = skips['conv3_1']
        
        # c, h, w = LH2.size()[-3:]
        b, c, h, w = LH2.size()
        LH2, HL2, HH2 = LH2.view(b, k, c, h, w).mean(dim=1), HL2.view(b, k, c, h, w).mean(dim=1), HH2.view(b, k, c, h,w).mean(dim=1)
        x_deconv2 = self.recon_block1(x1, LH2, HL2, HH2, original2)

        LH3, HL3, HH3 = skips['pool2']
        c, h, w = skips['conv2_1'].size()[-3:]
#        original3 = skips['conv2_1'].view(8, 3, c, h, w).mean(dim=1)
        # c, h, w = LH3.size()[-3:]
        b, c, h, w = LH3.size()
        LH3, HL3, HH3 = LH3.view(b, k, c, h, w).mean(dim=1), HL3.view(b, k, c, h, w).mean(dim=1), HH3.view(b, k, c, h,w).mean(dim=1)
        x_deconv4 = self.recon_block1(x3, LH3, HL3, HH3, original2)
        x5 = self.Upsample(x4+x_deconv2)
        x6 = self.Conv3(x5+x_deconv4)

        x7 = self.Upsample(x6)
        x8 = self.Conv4(x7)

        x9 = self.Conv5(x8)

        return x9

class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)