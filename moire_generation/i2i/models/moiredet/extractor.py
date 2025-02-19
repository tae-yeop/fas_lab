import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import resnet18, conv1x1, FPEM_FFM, UpScale
from performer_pytorch import Performer

# MoireDet
class MoireDetExtractor(nn.Module):
    def __init__(self, kernel_size=5, fpem_repeat=4, output_channel=4, repeat_times=3, 
                 channels=128, performer_output_channel=126):
        super().__init__()

        self.output_channel = output_channel
        self.kernel_size = kernel_size
        backbone_out = [64, 128, 256, 512]
        moire_ouput_channel = kernel_size ** 2 * output_channel
        
        self.moire_backbone = resnet18(pretrained=False, replace_stride_with_dilation=[True, True, False])
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=(kernel_size - 1) // 2)
        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[-1], performer_output_channel, (1, 1)),
            nn.BatchNorm2d(performer_output_channel)
        )

        self.attention_backbone = resnet18(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat, channels=channels, output_channel=1)
        
        

        self.with_pos_embedding = True
        nhead = 2
        performer_output_channel = performer_output_channel + 2

        self.performer = Performer(dim=performer_output_channel, depth=3, dim_head=64, heads=nhead,
                                   causal=True)
        self.upscale = UpScale(performer_output_channel * 2 - 2, 128, 32, 4, 4)
        
        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, output_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, output_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(output_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 3),
        )

        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1, max_len).repeat(1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len, 1).repeat(1, 1, max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len, max_len)

        
    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()
        
        moire_fea = self.moire_backbone(x)[-1]
        print('moire_fea', moire_fea.shape) # 1, 512, 32, 32
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size() #  1 126 32 32
        print('B, C, MH, MW', B, C, MH, MW)
        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea) # [1, 128, 64, 64])
        print('attention_fea', attention_fea.shape)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear', align_corners=True) # [1, 128, 64, 64])
        print('attention_fea2', attention.shape)
        moire_fea = moire_fea * attention
        B, C, H, W = moire_fea.size()

        dst_size = (H // 2, W // 3)
        min_trans_fea = F.interpolate(moire_fea, size=dst_size,
                                      mode='bilinear', align_corners=True)

        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        min_trans_fea = torch.cat([min_trans_fea, pos.repeat(B, 1, 1, 1)],
                                  dim=1)

        B, C, minH, minW = min_trans_fea.size()
        min_trans_fea = min_trans_fea.view(B, C, -1).permute(2, 0, 1).contiguous()


        min_trans_fea = self.performer(min_trans_fea).permute(1, 2, 0).contiguous().view(B, -1, minH, minW)
        min_trans_fea = F.interpolate(min_trans_fea, size=(H, W), mode='bicubic', align_corners=True)
        trans_fea = torch.cat([moire_fea, min_trans_fea], dim=1)
        trans_fea = self.upscale(trans_fea)
        _, _, h, w = trans_fea.size()
        
        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),
                                      mode='bicubic', align_corners=True)
        
        detail_conv = self.detail_conv(trans_fea)
        detail_conv = detail_conv.view(B, 1, self.output_channel, self.kernel_size ** 2, img_H, img_W)
        unfold_img = self.unfold(x) # x is input image
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)
        detail_conv = torch.softmax(detail_conv, dim=3)

        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.output_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = detail_bias.view(B, self.output_channel * 3, img_H, img_W)

        detail_fea = detail_fea + detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        return moire_density

    
class DnCNNExtractor(nn.Module):
    # https://github.com/cszn/DnCNN
    def __init__(self, ):
        super().__init__()
        ...
    def forward(self, x):
        ...
        # noise = self.model(x)
        # return noise


def get_wav_two(in_channels, pool=True):
    ...

class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        

class WvEncoder():
    def __init__(self):
        super().__init__()
        self.conv1 = ...

class WvDecoder():
    ...

    
class WaveletExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WvEncoder()
        self.decoder = WvDecoder()
        self.fusion = ...

if __name__ == '__main__':
    model = MoireDetExtractor()
    out = model(torch.randn(1, 3, 512, 512))
    print(out.shape)