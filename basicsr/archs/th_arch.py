from xml.etree.ElementInclude import include
import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer


class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p):
        super(DSConv, self).__init__()
        self.dsc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, s, p, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.dsc(x)


class MCSALayer(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat):
        super(MCSALayer, self).__init__()
        self.mcsa = nn.Sequential(
            DSConv(num_feat * 4, num_feat * 4, 3, 1, 1), nn.ReLU(inplace=True),
            DSConv(num_feat * 4, num_feat * 4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0, bias=False), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        y = self.mcsa(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class DWConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_size=7, stride=1, padding=3):
        super(DWConv, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, k_size, stride=stride, padding=padding, groups=in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        return self.body(x)


class ResU(nn.Module):

    def __init__(self, num_feat, k_size=7, res_scale=1):
        super(ResU, self).__init__()
        self.res_scale = res_scale
        self.body = DWConv(num_feat, num_feat, k_size=7, stride=1, padding=(k_size - 1) // 2)

    def forward(self, x):
        out = self.body(x)
        return out * self.res_scale + x


class THB(nn.Module):

    def __init__(self, num_feat, num_res=10, k_size=7, squeeze_factor=16, dst_feat=16, res_scale=1):
        super(THB, self).__init__()
        self.thb = []
        self.dst = []
        self.num_res = num_res
        for i in range(num_res):
            self.thb.append(ResU(num_feat, res_scale=res_scale, k_size=k_size))
            self.dst.append(nn.Conv2d(num_feat, dst_feat, 1, 1, 0, bias=False))
        self.thb = nn.ModuleList(self.thb)
        self.dst = nn.ModuleList(self.dst)

        self.att = ChannelAttention(num_feat + (num_res - 1) * dst_feat, squeeze_factor=16)
        self.cc = nn.Conv2d(num_feat + (num_res - 1) * dst_feat, num_feat, 1, 1, 0, bias=False)

    def forward(self, x):
        res = x
        dst = []
        for i in range(self.num_res - 1):
            x = self.thb[i](x)
            dst.append(self.dst[i](x))
        dst.append(self.thb[-1](x))
        x = self.att(torch.cat(dst, 1))
        x = self.cc(x)

        return res + x


@ARCH_REGISTRY.register()
class TH(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 k_size=7,
                 num_feat=64,
                 dst_feat=16,
                 num_res=10,
                 num_block=20,
                 squeeze_factor=16,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TH, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.sf = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.ModuleList([
            THB(num_feat, num_res, k_size=k_size, squeeze_factor=squeeze_factor, dst_feat=dst_feat, res_scale=res_scale)
            for i in range(num_block)
        ])
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)

        self.upsample = Upsample(upscale, num_feat)
        self.tail = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, k_size, stride=1, padding=(k_size - 1) // 2, groups=num_feat),
            nn.Conv2d(num_feat, num_out_ch, 1, 1, 0, bias=False))

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.sf(x)
        res = self.body[0](x)
        for b in self.body[1:]:
            res = b(res + x)
        res = self.conv_after_body(res + x)
        res = self.upsample(res + x)
        x = self.tail(res)

        x = x / self.img_range + self.mean

        return x
