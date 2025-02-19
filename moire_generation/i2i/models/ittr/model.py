import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# classes

class HPB(nn.Module):
    """ Hybrid Perception Block """

    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        attn_height_top_k = 16,
        attn_width_top_k = 16,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

        self.attn = DPSA(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            height_top_k = attn_height_top_k,
            width_top_k = attn_width_top_k,
            dropout = attn_dropout
        )

        self.dwconv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)
        self.attn_parallel_combine_out = nn.Conv2d(dim * 2, dim, 1)

        ff_inner_dim = dim * ff_mult

        self.ff = nn.Sequential(
            nn.Conv2d(dim, ff_inner_dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(nn.Sequential(
                nn.Conv2d(ff_inner_dim, ff_inner_dim, 3, padding = 1, groups = ff_inner_dim),
                nn.InstanceNorm2d(ff_inner_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout)
            )),
            nn.Conv2d(ff_inner_dim, dim, 1),
            nn.InstanceNorm2d(dim) # <--------------------------- 수정
        )

        # print('ff_inner_dim', ff_inner_dim)
    def forward(self, x):
        attn_branch_out = self.attn(x)
        conv_branch_out = self.dwconv(x)

        concatted_branches = torch.cat((attn_branch_out, conv_branch_out), dim = 1)
        attn_out = self.attn_parallel_combine_out(concatted_branches) + x
        # print('attn_out.shape', attn_out.shape) #e([1, 128, 16, 16])
        return self.ff(attn_out)

class DPSA(nn.Module):
    """ Dual-pruned Self-attention Block """

    def __init__(
        self,
        dim,
        height_top_k = 16,
        width_top_k = 16,
        dim_head = 32,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 16, 16
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        #print('q, k, v', q.shape, k.shape, v.shape) # torch.Size([1, 256, 16, 16]) torch.Size([1, 256, 16, 16]) torch.Size([1, 256, 16, 16])   
        # 256 = 32 * 8
        # fold out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = self.heads), (q, k, v))
        #print('q, k, v', q.shape, k.shape, v.shape) # torch.Size([8, 32, 16, 16]) torch.Size([8, 32, 16, 16]) torch.Size([8, 32, 16, 16])  
        # they used l2 normalized queries and keys, cosine sim attention basically

        q, k = map(l2norm, (q, k))
        #print('q, k, v', q.shape, k.shape, v.shape) # torch.Size([8, 32, 16, 16]) torch.Size([8, 32, 16, 16]) torch.Size([8, 32, 16, 16])
        # calculate whether to select and rank along height and width

        q, k, v = map(lambda t: rearrange(t, '(b h) c x y -> (b h) x y c', h=self.heads, x = h, y = w), (q, k, v)) # <-------- 추가

        need_height_select_and_rank = self.height_top_k < h
        need_width_select_and_rank = self.width_top_k < w

        # select and rank keys / values, probing with query (reduced along height and width) and keys reduced along row and column respectively

        if need_width_select_and_rank or need_height_select_and_rank:
            q_probe = reduce(q, 'b h w d -> b d', 'sum')

        # gather along height, then width

        if need_height_select_and_rank:
            k_height = reduce(k, 'b h w d -> b h d', 'sum')

            top_h_indices = einsum('b d, b h d -> b h', q_probe, k_height).topk(k = self.height_top_k, dim = -1).indices

            top_h_indices = repeat(top_h_indices, 'b h -> b h w d', d = self.dim_head, w = k.shape[-2])

            k, v = map(lambda t: t.gather(1, top_h_indices), (k, v)) # first gather across height

        if need_width_select_and_rank:
            k_width = reduce(k, 'b h w d -> b w d', 'sum')

            top_w_indices = einsum('b d, b w d -> b w', q_probe, k_width).topk(k = self.width_top_k, dim = -1).indices

            top_w_indices = repeat(top_w_indices, 'b w -> b h w d', d = self.dim_head, h = k.shape[1])

            k, v = map(lambda t: t.gather(2, top_w_indices), (k, v)) # then gather along width

        # select the appropriate keys and values

        q, k, v = map(lambda t: rearrange(t, 'b ... d -> b (...) d'), (q, k, v))
        # torch.Size([8, 32*16, 16]) torch.Size([8, 32*16, 16]) torch.Size([8, 32*16, 16])
        # cosine similarities

        sim = einsum('b i d, b j d -> b i j', q, k)
        # torch.Size([8, 32*16, 32*16])

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate out
        # attn의 [b, i, j]에서 i가 512이어서
        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge heads and combine out

        #print('out', out.shape) # (1, 128, 16, 16) => ([8, 512, 16]) (512가 아니라 256여야 할텐데)
        #print('h, w, self.heads', h, w, self.heads) #  16 16 8
        # h는 8이어야 할텐데
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = h, y = w, h = self.heads)
        return self.to_out(out)
        # [1, (),  16, 16]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=None):
        super().__init__()
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample_layer(x)
        x = self.conv(x)
        x = self.ins(x)
        x = self.gelu(x)
        return x

class ITTR(nn.Module):
    def __init__(self, channels, num_hpbs):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            BasicBlock(3, channels, kernel_size=7, stride=1, padding=3),
            BasicBlock(channels, channels, kernel_size=3, stride=2, padding=1),
            BasicBlock(channels, channels, kernel_size=3, stride=2, padding=1)
        )

        # HPBs
        self.hpbs = nn.Sequential(*[HPB(dim=channels) for _ in range(num_hpbs)])

        # Head
        self.head = nn.Sequential(
            BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1, upsample=2),
            BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1, upsample=2),
            nn.Conv2d(channels, 3, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        x = self.stem(x)
        # print('x.shape', x.shape)
        x = self.hpbs(x)
        # print('x.shape', x.shape)
        x = self.head(x)
        return x

# class Translator(nn.Module):
#     def __init__(self, channels, num_hpbs):
#         super().__init__()
#         self.channels = channels

#         # Stem
#         self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7, stride=1, padding=3)
#         self.ins1 = nn.InstanceNorm2d(self.channels)
#         self.gelu1 = nn.GELU()

#         self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)
#         self.ins2 = nn.InstanceNorm2d(self.channels)
#         self.gelu2 = nn.GELU()

#         self.conv3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
#         self.ins2 = nn.InstanceNorm2d(self.channels)
#         self.gelu3 = nn.GELU()

#         # HPBs
#         self.num_hpbs = num_hpbs
#         hbps_list = [HPB(dim=self.channels) for i in range(num_hpbs)]
#         self.hpbs = nn.Sequential(**hbps_list)

#         # Head
#         self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_head1 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
#         self.ins_head1 = nn.InstanceNorm2d(self.channels)
#         self.glue_head1 = nn.GELU()

#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
#         self.conv_head2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
#         self.ins_head2 = nn.InstanceNorm2d(self.channels)
#         self.glue_head2 = nn.GELU()

#         self.out = nn.Conv2d(self.channels, 3, kernel_size=7, stride=1, padding=3)


#     def forward(self, x):


import warnings
import sys
import traceback

# 경고를 발생시킬 때마다 스택 추적을 출력하도록 설정
warnings.filterwarnings("always")  # 모든 경고를 항상 표시
warnings.simplefilter("default")  # 기본 필터 설정 사용, 변경 가능


import warnings
import traceback
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(file=log)

warnings.showwarning = warn_with_traceback


if __name__ == '__main__':
    # dpsa = DPSA(dim=128).cuda()
    # i = torch.randn(1, 128, 16, 16).cuda()

    # print(dpsa(i).shape) # torch.Size([1, 128, 16, 16])


    # hpb = HPB(dim=128).cuda()
    # hpb2 = HPB(dim=128).cuda()

    # out = hpb(i)
    # print('out.shape', out.shape)
    # out2 = hpb2(out)
    # print('out2.shape', out2.shape)

    # print(nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=3)(torch.randn(1, 3, 512, 512)).shape)

    t = IITransformer(256, 3).cuda()
    #print(t(torch.randn(1, 3, 512, 512)).shape)
    print(t(torch.randn(1, 3, 256, 256).cuda()).shape)
