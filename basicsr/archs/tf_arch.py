from bisect import bisect
import torch
from torch import nn as nn
import math

from basicsr.utils.registry import ARCH_REGISTRY


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class TFB_Residual(nn.Module):

    def __init__(self, num_feat, k_size=3):
        super(TFB_Residual, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1))

    def forward(self, x):
        out = self.body(x)
        return torch.cat([x, out + x], 1)


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


class TFB(nn.Module):

    def __init__(self, num_feat, n_res):
        super(TFB, self).__init__()
        self.body = nn.Sequential(*[TFB_Residual(num_feat * 2**i) for i in range(n_res)])
        self.llf = nn.Conv2d(num_feat * 2**n_res, num_feat, 1, padding=0, stride=1, bias=False)
        self.ca = MCSALayer(num_feat)
        # self.g = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 1, padding=0, stride=1, bias=False))

    def forward(self, x):
        out = self.body(x)
        out = self.llf(out)
        out = self.ca(out)
        out += x
        return out


class TFG(nn.Module):

    def __init__(self, num_feat, n_res, n_block):
        super(TFG, self).__init__()
        self.body = nn.Sequential(*[TFB(num_feat, n_res) for i in range(n_block)])

    def forward(self, x):
        return self.body(x) + x


@ARCH_REGISTRY.register()
class TF(nn.Module):
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
                 num_feat=64,
                 num_cat_feat=128,
                 num_group=10,
                 num_block=5,
                 num_res=3,
                 num_dil=16,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TF, self).__init__()

        self.num_group = num_group
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.sfe = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        self.group = nn.ModuleList()
        self.dil = nn.ModuleList()
        for i in range(num_group):
            self.group.append(TFG(num_feat, num_res, num_block))
            self.dil.append(nn.Conv2d(num_feat, num_dil, 3, 1, 1, bias=False))

        self.conv_after_body = nn.Sequential(
            nn.Conv2d(num_feat + (num_group - 1) * num_dil, num_cat_feat, 3, 1, 1, bias=False))
        self.upsample = Upsample(upscale, num_cat_feat)
        self.tail = nn.Conv2d(num_cat_feat + num_in_ch, num_out_ch, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        u = self.up(x)
        x = self.head(x)
        x = self.sfe(x)
        out = x

        d_out = []
        out = self.group[0](x)
        d_out.append(self.dil[0](out))
        for i in range(self.num_group - 2):
            out = self.group[i + 1](out + x)
            d_out.append(self.dil[i + 1](out))
        d_out.append(self.group[self.num_group - 1](out + x))
        out = self.conv_after_body(torch.cat(d_out, 1))

        x = self.upsample(out)

        x = self.tail(torch.cat(u, x))
        x = x / self.img_range + self.mean

        return x
