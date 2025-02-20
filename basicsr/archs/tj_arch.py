import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer


torch.backends.cudnn.benchmark=True

class UDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.ReLU(True))

    def forward(self, x):
        return self.maxpool_conv(x)


class UUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.ReLU(True))

    def forward(self, x1, x2):
        x1 = self.up(x1)  # input is CHW

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, num_feat, n_step=3):
        super(UNet, self).__init__()
        self.n_step = n_step

        self.head = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(True))
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(n_step):
            self.down.append(UDown(num_feat * 2**i, num_feat * 2**(i + 1)))
            self.up.append(UUp(num_feat * 2**(i + 1), num_feat * 2**i))

        self.tail = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.head(x)
        res = [out]
        for i in range(self.n_step - 1):
            out = self.down[i](out)
            res.append(out)
        out = self.down[-1](out)
        for i in range(self.n_step)[::-1]:
            out = self.up[i](out, res[i])
        out = self.tail(x)
        return out


class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_size=7, stride=1, padding=3):
        super(DSConv, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k_size, stride=stride, padding=padding, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        return self.body(x)


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


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DSConv(num_feat, num_feat), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x)
        return res * self.res_scale + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = DSConv(num_feat, num_feat)

    def forward(self, x):
        x, ua = x
        res = self.conv(self.residual_group(x)) * ua
        return (res + x, ua)


@ARCH_REGISTRY.register()
class TJ(nn.Module):
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
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TJ, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = DSConv(num_feat, num_out_ch)
        self.ua = UNet(num_feat)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        ua = self.ua(x)
        res = self.conv_after_body(self.body((x, ua))[0])
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
