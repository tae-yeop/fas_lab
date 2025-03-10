import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet6', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d','conv3x3','conv1x1']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) < len(layers) -1 :
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()

        if len(layers) >= 1:
            self.layer1 = self._make_layer(block, 64, layers[0])
        if len(layers) >= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        if len(layers) >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if len(layers) >= 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)#,model_dir='pretrain_model'
        model.load_state_dict(state_dict, strict=False)
        print('load pretrained models from imagenet')
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class FPEM(nn.Module):
    def __init__(self, in_channel=128):
        super().__init__()
        self.add_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.add_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.add_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.add_down_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.add_down_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.add_down_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, c2, c3, c4, c5):
        c4 = self.add_up_1(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        c3 = self.add_up_2(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c2 = self.add_up_3(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))

        c3 = self.add_down_1(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))
        c4 = self.add_down_2(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c5 = self.add_down_3(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        return c2, c3, c4, c5

class FPEM_FFM(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        channels = kwargs.get('channels', 128)
        ouput_channel = kwargs.get('output_channel', 128)

        self.conv_c2 = nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=channels, kernel_size=1)
        self.conv_c3 = nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=channels, kernel_size=1)
        self.conv_c4 = nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=channels, kernel_size=1)
        self.conv_c5 = nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=channels, kernel_size=1)

        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(channels))
        self.out_conv = nn.Conv2d(in_channels=512, out_channels=ouput_channel, kernel_size=1)

        # print('fpem_repeat, channels, ouput_channel', fpem_repeat, channels, ouput_channel)
    def forward(self, x):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        c5 = self.conv_c5(c5)

        c2_ffm = c2
        c3_ffm = c3
        c4_ffm = c4
        c5_ffm = c5

        for fpem in self.fpems:
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            c2_ffm += c2
            c3_ffm += c3
            c4_ffm += c4
            c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        return y


class UpScale(nn.Module):
    def __init__(self, inchannels=128, feature_channels=128, outchannels=32,
                 fx=4, fy=4):
        super().__init__()

        self.fx = fx
        self.fy = fy
        self.outchannels = outchannels

        self.mask = nn.Sequential(
            nn.Conv2d(inchannels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, fx * fy * 9, 1, padding=0))

        self.feafusion = nn.Sequential(
            nn.Conv2d(inchannels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, outchannels, 1, padding=0))

    def forward(self, x):
        N, _, H, W = x.shape

        fea = self.feafusion(x)

        mask = self.mask(x)
        mask = mask.view(N, 1, 9, self.fy, self.fx, H, W)
        mask = torch.softmax(mask, dim=2)

        fea = F.unfold(fea, [3, 3], padding=1)
        fea = fea.view(N, self.outchannels, 9, 1, 1, H, W)

        fea = torch.sum(mask * fea, dim=2)
        fea = fea.permute(0, 1, 4, 2, 5, 3)
        return fea.reshape(N, self.outchannels, self.fy * H, self.fx * W)