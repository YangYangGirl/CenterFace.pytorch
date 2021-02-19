from __future__ import absolute_import, division, print_function

import math

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init

import sys
sys.path.append('./')
from dcn_v2 import DCN
from ..module.conv import ConvModule
from ..module.init_weights import xavier_init

class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear')

        # build outputs
        outs = [
            # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            laterals[i] for i in range(used_backbone_levels)
        ]
        return tuple(outs)


class PAN(FPN):
    """Path Aggregation Network for Instance Segmentation.
    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PAN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, conv_cfg, norm_cfg, activation)
        self.init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            laterals[i] for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            prev_shape = inter_outs[i + 1].shape[2:]
            inter_outs[i + 1] += F.interpolate(inter_outs[i], size=prev_shape, mode='bilinear')

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            inter_outs[i] for i in range(1, used_backbone_levels)
        ])
        return tuple(outs)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

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


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=True, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p6_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # self.p5_to_p6 = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            #     MaxPool2dStaticSamePadding(3, 2)
            # )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.convyy1 = nn.Conv2d(960, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.convyy2 = nn.Conv2d(160, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.convyy3 = nn.Conv2d(40, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.convyy4 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False)

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """



        """
        illustration of a minimal bifpn unit
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|                ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        # import pdb; pdb.set_trace()
        if self.first_time:
            p3, p4, p5, p6 = inputs

            # p6_in = self.p5_to_p6(p5)
            # p7_in = self.p6_to_p7(p6)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            p6_in = self.p6_down_channel(p6)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in = inputs
            # p3_in, p4_in, p5_in, p6_in = self.convyy4(p3_in), self.convyy3(p4_in), self.convyy2(p5_in), self.convyy1(p6_in),
            # p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # # Weights for P6_0 and P7_0 to P6_1
        # p6_w1 = self.p6_w1_relu(self.p6_w1)
        # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # # Connections for P6_0 and P7_0 to P6_1 respectively
        # p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_in)))
        # p5_up = self.conv5_up(self.swish(weight[0] * p5_in + self.convyy1(weight[1] * self.p5_upsample(p6_in))))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        # import pdb; pdb.set_trace()
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in) + weight[1] * self.p4_upsample(p5_up))
        # p4_up = self.conv4_up(self.convyy3(self.swish(weight[0] * p4_in)) + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        # import pdb; pdb.set_trace()
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in) + weight[1] * self.p3_upsample(p4_up))
        # p3_out = self.conv3_up(self.convyy4(self.swish(weight[0] * p3_in)) + weight[1] * self.p3_upsample(p4_up))


        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively

        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_in + weight[2] * self.p6_downsample(p5_in)))

        # # Weights for P7_0 and P6_2 to P7_2
        # p7_w2 = self.p7_w2_relu(self.p7_w2)
        # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # # Connections for P7_0 and P6_2 to P7_2
        # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        # return p3_out, p4_out, p5_out, p6_out, p7_out
        # import pdb; pdb.set_trace()
        return p3_out, p4_out, p5_out, p6_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5, p6 = inputs

            # p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            p6_in = self.p6_down_channel(p6)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            # p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # # Connections for P6_0 and P7_0 to P6_1 respectively
            # p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
            
            # Connections for P5_0 and P6_1 to P5_1 respectively
             p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # if self.use_p8:
        #     # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
        #     p7_out = self.conv7_down(
        #         self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

        #     # Connections for P8_0 and P7_2 to P8_2
        #     p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

        #     return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        # else:
        #     # Connections for P7_0 and P6_2 to P7_2
        #     p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            # return p3_out, p4_out, p5_out, p6_out, p7_out
        
        return p3_out, p4_out, p5_out, p6_out

# borrow from yolov4
class mish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        in_size = int(in_size)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        in_size = int(in_size)
        out_size = int(out_size)
        expand_size = int(expand_size)
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
        

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


class MobileNetV3(nn.Module):
    def __init__(self, heads, head_conv, n_class=1000, input_size=224, width_mult=1., final_kernel=1, activation='hswish', if_bifpn=False, is_pan=True, is_simple=True):
        super(MobileNetV3, self).__init__()
        self.activation = activation
        self.if_bifpn = if_bifpn
        self.is_pan = is_pan
        self.is_simple = is_simple
        self.conv1 = nn.Conv2d(3, int(16 * width_mult), kernel_size=3, stride=2, padding=1, bias=False)
        self.convyy = nn.Conv2d(int(16 * width_mult), int(16 * width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16 * width_mult))

        if self.activation == 'mish':
            self.hs1 = mish()
            self.hs2 = mish()
            self.bneck2 = nn.Sequential(
            Block(3, 40 * width_mult, 240 * width_mult, 80 * width_mult, mish(), None, 2),
            Block(3, 80 * width_mult, 200 * width_mult, 80 * width_mult, mish(), None, 1),
            Block(3, 80 * width_mult, 184 * width_mult, 80 * width_mult, mish(), None, 1),
            Block(3, 80 * width_mult, 184 * width_mult, 80 * width_mult, mish(), None, 1),
            Block(3, 80 * width_mult, 480 * width_mult, 112 * width_mult, mish(), SeModule(112 * width_mult), 1),
            Block(3, 112 * width_mult, 672 * width_mult, 112 * width_mult, mish(), SeModule(112 * width_mult), 1),
            Block(5, 112 * width_mult, 672 * width_mult, 160 * width_mult, mish(), SeModule(160 * width_mult), 1),
            )
            self.bneck3 = nn.Sequential(
                Block(5, 160 * width_mult, 672, 160 * width_mult, mish(), SeModule(160 * width_mult), 2),
                Block(5, 160 * width_mult, 960, 160 * width_mult, mish(), SeModule(160 * width_mult), 1),
            )
        elif self.activation == 'hswish':
            self.hs1 = hswish()
            self.hs2 = hswish()
            self.bneck2 = nn.Sequential(
            Block(3, 40 * width_mult, 240 * width_mult, 80 * width_mult, hswish(), None, 2),
            Block(3, 80 * width_mult, 200 * width_mult, 80 * width_mult, hswish(), None, 1),
            Block(3, 80 * width_mult, 184 * width_mult, 80 * width_mult, hswish(), None, 1),
            Block(3, 80 * width_mult, 184 * width_mult, 80 * width_mult, hswish(), None, 1),
            Block(3, 80 * width_mult, 480 * width_mult, 112 * width_mult, hswish(), SeModule(112 * width_mult), 1),
            Block(3, 112 * width_mult, 672 * width_mult, 112 * width_mult, hswish(), SeModule(112 * width_mult), 1),
            Block(5, 112 * width_mult, 672 * width_mult, 160 * width_mult, hswish(), SeModule(160 * width_mult), 1),
            )
            self.bneck3 = nn.Sequential(
                Block(5, 160 * width_mult, 672 * width_mult, 160 * width_mult, hswish(), SeModule(160 * width_mult), 2),
                Block(5, 160 * width_mult, 960 * width_mult, 160 * width_mult, hswish(), SeModule(160 * width_mult), 1),
            )

        self.bneck0 = nn.Sequential(
            #kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
            Block(3, 16 * width_mult, 16 * width_mult, 16 * width_mult, nn.ReLU(inplace=True), None, 1),
            Block(3, 16 * width_mult, 64 * width_mult, 24 * width_mult, nn.ReLU(inplace=True), None, 2),
            Block(3, 24 * width_mult, 72 * width_mult, 24 * width_mult, nn.ReLU(inplace=True), None, 1),
        )

        self.bneck1 = nn.Sequential(
            Block(5, 24 * width_mult, 72 * width_mult, 40 * width_mult, nn.ReLU(inplace=True), SeModule(40 * width_mult), 2),
            Block(5, 40 * width_mult, 120 * width_mult, 40 * width_mult, nn.ReLU(inplace=True), SeModule(40 * width_mult), 1),
            Block(5, 40 * width_mult, 120 * width_mult, 40 * width_mult, nn.ReLU(inplace=True), SeModule(40 * width_mult), 1),
        )
        
        self.conv2 = nn.Conv2d(int(160 * width_mult), int(960 * width_mult), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(960 * width_mult))
        
        channel_list = [int(c * width_mult) for c in [24, 40, 160, 960]]
        self.ida_up = IDAUp(int(24 * width_mult), channel_list,
                            [2 ** i for i in range(4)])

        if self.if_bifpn:
            self.bifpn = BiFPN(int(24 * width_mult), channel_list)
        if self.is_pan:
            self.pan = PAN(channel_list, int(24 * width_mult), 4)

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(int(24 * width_mult), head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(int(24 * width_mult), classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out0 = self.bneck0(out)
        out1 = self.bneck1(out0)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        out3 = self.hs2(self.bn2(self.conv2(out3)))

        out = [out0, out1, out2, out3]

        y = []
        for i in range(4):
            y.append(out[i].clone())

        if self.is_simple:
            z = self.pan(y)
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(z[0])
        elif self.if_bifpn:
            z = self.bifpn(y)
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(z[0])
        elif self.is_pan:
            z = self.pan(y)
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(z[0])
        else:
            self.ida_up(y, 0, len(y))
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(y[-1])

        return [ret]
        
    
def get_mobilev3_pose_net(num_layers, heads, head_conv=160):
  model = MobileNetV3(heads, head_conv, width_mult=1.0)
#   model = MobileNetV3(heads, head_conv, width_mult=0.75)
  
  return model
