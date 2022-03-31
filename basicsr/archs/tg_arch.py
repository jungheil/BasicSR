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


class ResU(nn.Module):

    def __init__(self, num_feat, k_size=3, res_scale=1):
        super(ResU, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1))

    def forward(self, x):
        out = self.body(x)
        return out * self.res_scale + x


class TGB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_res=10, squeeze_factor=16, res_scale=1):
        super(TGB, self).__init__()
        self.tgb = [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(num_res):
            self.tgb.append(ResU(num_feat, res_scale=res_scale))
        self.tgb = nn.Sequential(*self.tgb)
        self.mcsa = MCSALayer(num_feat)
        self.tail = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 1, padding=0, stride=1, bias=False),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        res = self.tgb(x)
        res = self.mcsa(res)
        res = self.tail(torch.cat([x, res], 1))
        return res + x


@ARCH_REGISTRY.register()
class TG(nn.Module):
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
                 num_res=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TG, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([TGB(num_feat, num_res, squeeze_factor, res_scale) for i in range(num_block)])
        self.conv_after_body = nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1, bias=False)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat + num_in_ch, num_out_ch, 3, 1, 1)
        self.ds1 = nn.Sequential(
            nn.MaxPool2d(2, 2), TGB(num_feat, num_res, squeeze_factor, res_scale),
            TGB(num_feat, num_res, squeeze_factor, res_scale))
        self.ds2 = nn.Sequential(
            nn.MaxPool2d(2, 2), TGB(num_feat, num_res, squeeze_factor, res_scale),
            TGB(num_feat, num_res, squeeze_factor, res_scale))
        self.us1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.us2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.usg = nn.Upsample(scale_factor=upscale, mode='bicubic', align_corners=False)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range

        xx = self.conv_first(x)
        res = self.body[0](xx)
        for b in self.body[1:]:
            res = b(res + xx)
        d1 = self.ds1(res)
        d2 = self.ds2(d1)
        d1 = self.us1(d1)
        d2 = self.us2(d2)
        res = self.conv_after_body(torch.cat([res, d1, d2], 1))
        res = self.upsample(res)
        x = self.conv_last(torch.cat([self.usg(x), res], 1))
        x = x / self.img_range + self.mean

        return x
