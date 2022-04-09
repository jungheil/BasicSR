import torch.nn as nn
from collections import OrderedDict
import torch

from basicsr.utils.registry import ARCH_REGISTRY
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
torch.backends.cudnn.benchmark = True


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc,
               out_nc,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True,
               pad_type='zero',
               norm_type=None,
               act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# contrast-aware channel attention module
class CCALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


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
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
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

    def __init__(self, num_feat, n_step=2, dltail=False):
        super(UNet, self).__init__()
        self.n_step = n_step
        self.dltail = dltail

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
        if not self.dltail:
            out = self.tail(x)
        return out


class Block(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale

        # self.body = nn.Sequential(
        #     nn.Conv2d(num_feat, num_feat, 7, 1, 3, groups=num_feat, bias=False),
        #     nn.InstanceNorm2d(num_feat, affine=True), nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0), nn.GELU(),
        #     nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0))
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 7, 1, 3, groups=num_feat, bias=False),
            nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0), nn.GELU(),
            nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0))
        # self.body = nn.Sequential(
        #     DepthWiseConv2dImplicitGEMM(num_feat, 7), nn.InstanceNorm2d(num_feat, affine=True),
        #     nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0), nn.GELU(), nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0))

    def forward(self, x):
        res = self.body(x)
        return res * self.res_scale + x


class IMDModule(nn.Module):

    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        distilled_channels = int(in_channels * distillation_rate)
        self.c1 = Block(in_channels)
        self.c2 = Block(in_channels)
        self.c3 = Block(in_channels)
        self.c4 = Block(in_channels)
        self.d1 = nn.Conv2d(in_channels, distilled_channels, 1, 1, 0)
        self.d2 = nn.Conv2d(in_channels, distilled_channels, 1, 1, 0)
        self.d3 = nn.Conv2d(in_channels, distilled_channels, 1, 1, 0)

        self.c5 = nn.Conv2d(distilled_channels * 3 + in_channels, in_channels, 1, 1, 0)
        self.am = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, input, ua):
        out_c1 = self.c1(input)
        distilled_c1 = self.d1(out_c1)
        out_c2 = self.c2(out_c1)
        distilled_c2 = self.d2(out_c2)
        out_c3 = self.c3(out_c2)
        distilled_c3 = self.d3(out_c3)
        out_c4 = self.c4(out_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) * self.am(ua) + input
        return out_fused


@ARCH_REGISTRY.register()
class TL(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, num_modules=6, num_out_ch=3, upscale=4):
        super(TL, self).__init__()
        nf = num_feat
        self.fea_conv = conv_layer(num_in_ch, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, nf, upscale_factor=upscale)
        self.am = UNet(nf, 2, True)
        self.tail = nn.Conv2d(nf, num_out_ch, 3, 1, 1)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        ua = self.am(out_fea)
        out_B1 = self.IMDB1(out_fea, ua)
        out_B2 = self.IMDB2(out_B1, ua)
        out_B3 = self.IMDB3(out_B2, ua)
        out_B4 = self.IMDB4(out_B3, ua)
        out_B5 = self.IMDB5(out_B4, ua)
        out_B6 = self.IMDB6(out_B5, ua)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        output = self.tail(output)
        return output