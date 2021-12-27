"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-27 03:13:17
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-27 03:18:15
"""
"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from builtins import isinstance

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.layers import MLGConv2d, MLGLinear

from .model_base import MLGBaseModel

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    padding=1,
    bias=False,
    ### unique parameters
    in_bit=16,
    w_bit=16,
    device=torch.device("cuda"),
):
    conv = MLGConv2d(
        in_planes,
        out_planes,
        3,
        stride=stride,
        padding=padding,
        bias=bias,
        in_bit=in_bit,
        w_bit=w_bit,
        device=device,
    )
    return conv


def conv1x1(
    in_planes,
    out_planes,
    stride=1,
    bias=False,
    ### unique parameters
    in_bit=16,
    w_bit=16,
    device=torch.device("cuda"),
):
    """1x1 convolution"""
    conv = MLGConv2d(
        in_planes,
        out_planes,
        1,
        stride=stride,
        padding=0,
        bias=bias,
        in_bit=in_bit,
        w_bit=w_bit,
        device=device,
    )
    return conv


def Linear(
    in_channel,
    out_channel,
    bias=False,
    ### unique parameters
    in_bit=16,
    w_bit=16,
    device=torch.device("cuda"),
):
    linear = MLGLinear(in_channel, out_channel, bias=False, in_bit=in_bit, w_bit=w_bit, device=device)
    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        ## unique parameters
        in_bit=16,
        w_bit=16,
        device=torch.device("cuda"),
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes, planes, stride=stride, padding=1, bias=False, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(
            planes, planes, stride=1, padding=1, bias=False, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    stride=stride,
                    bias=False,
                    in_bit=in_bit,
                    w_bit=w_bit,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        ## unique parameters
        in_bit=16,
        w_bit=16,
        device=torch.device("cuda"),
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(
            in_planes,
            planes,
            stride=1,
            bias=False,
            in_bit=in_bit,
            w_bit=w_bit,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(
            planes, planes, stride=stride, padding=1, bias=False, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            stride=1,
            bias=False,
            in_bit=in_bit,
            w_bit=w_bit,
            device=device,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    stride=stride,
                    bias=False,
                    in_bit=in_bit,
                    w_bit=w_bit,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(MLGBaseModel):
    _conv = (MLGConv2d,)
    _linear = (MLGLinear,)
    _conv_linear = (MLGConv2d, MLGLinear)

    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        ### unique parameters
        in_channels=3,
        in_bit=16,
        w_bit=16,
        device=torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.device = device

        self.in_planes = 64
        self.conv1 = conv3x3(
            in_channels, 64, stride=1, padding=1, bias=False, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, in_bit=in_bit, w_bit=w_bit, device=device
        )
        self.linear = Linear(
            512 * block.expansion, num_classes, bias=False, in_bit=in_bit, w_bit=w_bit, device=device
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.reset_parameters()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        ### unique parameters
        in_bit=16,
        w_bit=16,
        device=torch.device("cuda"),
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, in_bit=in_bit, w_bit=w_bit, device=device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
