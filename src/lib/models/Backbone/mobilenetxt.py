"""
non-official PyTorch implementation of MobileNeXt from paper:
Rethinking Bottleneck Structure for Efficient Mobile Network Design
https://arxiv.org/abs/2007.02269

modified from mobilenetv2 torchvision implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py

"""

import math
import torch
from torch import nn


import sys
sys.path.append('./')
from ..networks.DCNv2.dcn_v2 import DCN

__all__ = ['MobileNeXt', 'mobilenext']



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier==1.0 else True
        self.identity_tensor_channels = int(round(inp*identity_tensor_multiplier))

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup /6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor= x[:,:self.identity_tensor_channels,:,:] + out[:,:self.identity_tensor_channels,:,:]
                out = torch.cat([identity_tensor, out[:,self.identity_tensor_channels:,:,:]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

            
class MobileNeXt(nn.Module):
    def __init__(
        self, heads, head_conv, 
        width_mult=1., 
        input_size=224, 
        identity_tensor_multiplier=1.0, 
        sand_glass_setting=None,
        round_nearest=8,
        block=None,
        norm_layer=None):
        """
        MobileNeXt main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            identity_tensor_multiplier(float): Identity tensor multiplier - reduce the number of element-wise additions in each block
            sand_glass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNeXt, self).__init__()

        self.inplanes = 64
        self.deconv_with_bias = False
        if block is None:
            block = SandGlass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                [2, 96,  1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 2, 1],
                # [2, 16,  1, 2],
                # [6, 24, 1, 1],
                # [6, 32, 3, 2],
                # [6, 64, 3, 2],
                # [6, 96, 4, 1],
                # [6, 160, 4, 2],
                # [6, 320, 2, 1],
                [6, self.last_channel / width_mult, 1, 1],
            ]

        self.ida_up = IDAUp(96, [96, 192, 384, 960],
                            [2 ** i for i in range(4)])
        # self.ida_up = IDAUp(24, [24, 40, 160, 960],
        #                     [2 ** i for i in range(4)])

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
            raise ValueError("sand_glass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sand_glass_setting))

        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, identity_tensor_multiplier=identity_tensor_multiplier, norm_layer=norm_layer))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        self.deconv_layers = self._make_deconv_layer(
            3,
            [head_conv, head_conv, head_conv],
            [4, 4, 4],
        )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.Sigmoid()
                )
            else:
                fc = nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True)
            # if 'hm' in head:
            #     fc.bias.data.fill_(-2.19)
            # else:
            #     nn.init.normal_(fc.weight, std=0.001)
            #     nn.init.constant_(fc.bias, 0)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
            assert num_layers == len(num_filters), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'
            assert num_layers == len(num_kernels), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

                planes = num_filters[i]
                fc = DCN(self.inplanes, planes, 
                        kernel_size=(3,3), stride=1,
                        padding=1, dilation=1, deformable_groups=1)
                # fc = nn.Conv2d(self.inplanes, planes,
                #         kernel_size=3, stride=1, 
                #         padding=1, dilation=1, bias=False)
                # fill_fc_weights(fc)
                up = nn.ConvTranspose2d(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias)
                fill_up_weights(up)

                layers.append(fc)
                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))
                layers.append(up)
                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))
                self.inplanes = planes

            return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # print(self.features)
        outs = []
        for f in self.features:
            x = f(x)
            outs.append(x)
            # print(x.shape)
        
        #0 torch.Size([6, 96, 200, 200]) 
        #1 torch.Size([6, 144, 200, 200]) 
        #2 torch.Size([6, 192, 100, 100])
        #3 torch.Size([6, 192, 100, 100])
        #4 torch.Size([6, 192, 100, 100])
        #5 torch.Size([6, 288, 50, 50])
        #6 torch.Size([6, 288, 50, 50])
        #7 torch.Size([6, 288, 50, 50])
        #8 torch.Size([6, 384, 50, 50])
        #9 torch.Size([6, 384, 50, 50])
        #10 torch.Size([6, 384, 50, 50])
        #11 torch.Size([6, 384, 50, 50])
        #12 torch.Size([6, 576, 25, 25])
        #13 torch.Size([6, 576, 25, 25])
        #14 torch.Size([6, 576, 25, 25])
        #15 torch.Size([6, 576, 25, 25])
        #16 torch.Size([6, 960, 25, 25])
        #17 torch.Size([6, 960, 25, 25])
        #18 torch.Size([6, 1280, 25, 25])

        # y = [outs[1], outs[4], outs[11], outs[18]]
        y = [outs[1], outs[4], outs[11], outs[18]]
        # y = [outs[1], outs[4], outs[11], outs[18]]

        self.ida_up(y, 0, len(y))

        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # print("==", x.shape)  #[6, 64, 25, 25]
        # x = self.deconv_layers(x)
        # print("!!", x.shape)  #[6, 64, 200, 200])
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(y[-1])
        return [ret]

    def forward(self, x):
        return self._forward_impl(x)


def mobilextnet_10(pretrained=True, **kwargs):
    model = MobileNeXt(width_mult=1.0)
    return model


def get_mobilext_net(num_layers, heads, head_conv=24):
    model = MobileNeXt(heads, head_conv, width_mult=1.0)
    return model

if __name__ == "__main__":
    model = MobileNeXt(width_mult=1.0, identity_tensor_multiplier=1.0)
    print(model)

    test_data = torch.rand(1, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())