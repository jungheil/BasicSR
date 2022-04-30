import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
import math

from basicsr.utils.registry import ARCH_REGISTRY

# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
torch.backends.cudnn.benchmark = True

global DEBUG
DEBUG = True

if DEBUG:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import utils

    tb_logger = SummaryWriter(log_dir='./V')


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1, padding=1):
    conv = nn.Conv2d(
        in_channels, out_channels * (upscale_factor**2), kernel_size, stride, padding, padding_mode='replicate')
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups,
    )


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


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type='zero',
    norm_type=None,
    act_type='relu',
):
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
        groups=groups,
    )
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
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
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


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, padding_mode='replicate')

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size, i=0):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.i = i

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # v = utils.make_grid(spat_att.transpose(1, 0), 8, normalize=True, scale_each=True, value_range=(0, 1))
        # utils.save_image(v, 'A{}.png'.format(self.i))
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            padding_mode='replicate')
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in),
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out


# contrast-aware channel attention module
class CCALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True, padding_mode='replicate'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True, padding_mode='replicate'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class GCLayer(nn.Module):

    def __init__(self, channel, reduction=16, ln=True):
        super(GCLayer, self).__init__()

        self.CM = nn.Sequential(nn.Conv2d(channel, 1, 1, padding=0, bias=True, padding_mode='replicate'), nn.Sigmoid())
        if ln:
            self.T = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True, padding_mode='replicate'),
                nn.LayerNorm([channel // reduction, 1, 1]),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True, padding_mode='replicate'),
            )
        else:
            self.T = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True, padding_mode='replicate'),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True, padding_mode='replicate'),
            )

    def forward(self, x):
        b, n, w, h = x.size()
        v = x.view(-1, n, w * h)
        k = self.CM(x).view(-1, 1, w * h)
        k = k.permute(0, 2, 1)
        y = torch.bmm(v, k).view(b, n, 1, 1)
        y = self.T(y) + x
        return y


def ALayer(type, channel, i=0):
    if type == 'C':
        return CCALayer(channel)
    elif type == 'G':
        return GCLayer(channel)
    elif type == 'B':
        return CBAM(n_channels_in=channel, reduction_ratio=2, kernel_size=3, i=i)
    elif type == 'O':
        return CoordAtt(channel, channel)
    elif type == 'E':
        return eca_layer(channel)
    elif type == 'N':
        return NONLocalBlock2D(channel)
    else:
        raise RuntimeError('alayer type?')


class Block(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, res_scale=1, k_size=3, dilation=1, conv='N'):
        super(Block, self).__init__()
        self.res_scale = res_scale
        padding = int((k_size - 1) / 2) * dilation

        # self.body = nn.Sequential(
        #     nn.Conv2d(num_feat, num_feat, 7, 1, 3, groups=num_feat, bias=False),
        #     nn.InstanceNorm2d(num_feat, affine=True), nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0), nn.GELU(),
        #     nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0))

        if conv == 'N':
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, k_size, 1, padding, dilation, padding_mode='replicate'),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
            )
        elif conv == 'T':
            self.body = nn.Sequential(
                nn.Conv2d(
                    num_feat,
                    num_feat,
                    k_size,
                    1,
                    padding,
                    dilation,
                    groups=num_feat,
                    bias=False,
                    padding_mode='replicate'),
                nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0, padding_mode='replicate'),
                nn.SiLU(),
                nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0, padding_mode='replicate'),
            )
        elif conv == 'D':
            self.body = nn.Sequential(
                nn.Conv2d(
                    num_feat,
                    num_feat,
                    k_size,
                    1,
                    padding,
                    dilation,
                    groups=num_feat,
                    bias=False,
                ),
                nn.Conv2d(num_feat, num_feat, 1, 1, 0, padding_mode='replicate'),
                nn.SiLU(),
            )
        else:
            raise RuntimeError('conv?')

    def forward(self, x):
        res = self.body(x)
        return res * self.res_scale + x


class IMDModule(nn.Module):

    def __init__(self, in_channels, dtl_channels, num_b=4, k_size=3, conv='N', al='0C', att_res=False, i=0):
        super(IMDModule, self).__init__()
        self.al = al
        self.num_b = num_b
        self.att_res = att_res
        self.c = nn.ModuleList([Block(in_channels, k_size=k_size, conv=conv) for _ in range(num_b)])
        self.d = nn.ModuleList(
            [nn.Conv2d(in_channels, dtl_channels, 1, 1, 0, padding_mode='replicate') for _ in range(num_b)])

        self.dtl_tail = nn.Conv2d(dtl_channels * self.num_b, in_channels, 1, 1, 0, padding_mode='replicate')
        # self.tailbone =

        if al[0] != '0':
            self.al0 = ALayer(al[0], in_channels, i)
        if al[1] != '0':
            self.al1 = ALayer(al[1], dtl_channels * self.num_b, i + 10)

    def forward(self, input):
        dtl = []
        out = input
        for i in range(self.num_b):
            out = self.c[i](out)
            dtl.append(self.d[i](out))
        d = torch.cat(dtl, dim=1)
        if self.al[0] != '0':
            if self.att_res:
                out = self.al0(out) + out
            else:
                out = self.al0(out)
        if self.al[1] != '0':
            d = self.al1(d)
        d = self.dtl_tail(d)
        out = out + input
        return (out, d)


@ARCH_REGISTRY.register()
class TZ(nn.Module):

    def __init__(
            self,
            num_in_ch=3,
            num_feat=64,
            num_dtl_c=16,
            num_modules=6,
            num_block=4,
            num_out_ch=3,
            k_size='333333',
            conv_type='NNNNNN',
            al='0C',
            bone_tail=False,
            att_res=False,
            upscale=2,
            img_range=255.0,
            rgb_mean=(0.33024667, 0.41541553, 0.42345934),
    ):
        super(TZ, self).__init__()
        nf = num_feat
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = img_range
        self.bone_tail = bone_tail
        self.fea_conv = nn.Conv2d(num_in_ch, nf, 3, 1, 1, padding_mode='replicate')
        # IMDBs
        self.G = nn.ModuleList()

        for i in range(num_modules):
            self.G.append(IMDModule(nf, num_dtl_c, num_block, int(k_size[i]), conv_type[i], al, att_res, i))

        if bone_tail:
            tail_f = num_dtl_c * num_block * num_modules + num_feat
        else:
            tail_f = num_dtl_c * num_block * num_modules
        # TODO zuowanshiyanhou,kaolv 3*3, tongdaoxiaojianbuyaoyong relu
        self.dc = nn.Conv2d(tail_f, nf, 1, 1, 0, padding_mode='replicate')

        self.fuse = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, padding_mode='replicate'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, padding_mode='replicate'),
        )
        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, nf, upscale_factor=upscale)
        self.tail = nn.Conv2d(nf, num_out_ch, 3, 1, 1, padding_mode='replicate')

    def forward(self, x, i=0):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.fea_conv(x)
        dtl = []
        if DEBUG:
            V = []
            Vd = []
        out, d = self.G[0](x)
        if DEBUG:
            Vd.append(utils.make_grid(d.transpose(0, 1), 8, normalize=True, scale_each=True))
            V.append(utils.make_grid(out.transpose(0, 1), 8, normalize=True, scale_each=True))

        dtl.append(d)
        for g in self.G[1:]:
            out, d = g(out)
            if DEBUG:
                Vd.append(utils.make_grid(d.transpose(0, 1), 8, normalize=True, scale_each=True))
                V.append(utils.make_grid(out.transpose(0, 1), 8, normalize=True, scale_each=True))
            dtl.append(d)
        if self.bone_tail:
            dtl.append(out)

        # if DEBUG:
        #     for i, v in enumerate(V):
        #         # utils.save_image(v, '{}.png'.format(i))
        #         # utils.save_image(Vd[i], 'd{}.png'.format(i))
        #         tb_logger.add_image('out{}'.format(i), v, 0)
        #         tb_logger.add_image('d{}'.format(i), Vd[i], 0)

        out = self.dc(torch.cat(dtl, dim=1))
        out_lr = self.fuse(out) + x
        output = self.upsampler(out_lr)
        output = self.tail(output)
        output = output / self.img_range + self.mean

        if DEBUG:
            # for name, param in self.named_parameters():
            #     if name == 'dc.0.weight':
            #         for i in range(6):
            #             for j in range(4):
            #                 tb_logger.add_histogram(
            #                     tag=name + '_data{},{}'.format(i, j),
            #                     values=param.data[:, i * 64 + j * 16:i * 64 + (j + 1) * 16, :, :],
            #                     global_step=1)
            for name, param in self.named_parameters():
                if name == 'dc.weight':
                    for i in range(4):
                        tb_logger.add_histogram(
                            tag=name + '_data{}'.format(i),
                            values=param.data[:, i * 128:(i + 1) * 128, :, :],
                            global_step=1)

        return output
