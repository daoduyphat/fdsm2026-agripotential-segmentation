import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, ratio=2, relu=True, dilation=1):
        super().__init__()
        init_channels = int(out_ch / ratio)
        new_channels = out_ch - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = GhostConv(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, relu=True,
                               dilation=dilation)
        self.conv2 = GhostConv(planes, planes, kernel_size=3, stride=1, padding=1, relu=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(1, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels=12, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        if input_channels != 3:
            old = resnet.conv1
            new_conv1 = nn.Conv2d(input_channels, old.out_channels,
                                  kernel_size=old.kernel_size, stride=old.stride,
                                  padding=old.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv1.weight[:, :3, :, :] = old.weight
                    if input_channels > 3:
                        mean = old.weight.mean(dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            new_conv1.weight[:, i:i + 1, :, :] = mean
            resnet.conv1 = new_conv1

        def _make_layer(block, inplanes, planes, blocks, stride=1, dilation=1):
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = []
            layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
            inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(inplanes, planes, dilation=dilation))
            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool

        self.layer1 = _make_layer(GhostBasicBlock, 64, 64, 3)
        self.layer2 = _make_layer(GhostBasicBlock, 64, 128, 4, stride=2)
        self.layer3 = _make_layer(GhostBasicBlock, 128, 256, 6, stride=1, dilation=2)
        self.layer4 = _make_layer(GhostBasicBlock, 256, 512, 3, stride=1, dilation=4)

        self.att1 = ECA(64)
        self.att2 = ECA(128)
        self.att3 = ECA(256)
        self.att4 = ECA(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.att1(self.layer1(x))
        x2 = self.att2(self.layer2(x1))
        x3 = self.att3(self.layer3(x2))
        x4 = self.att4(self.layer4(x3))
        return x1, x4


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rates = [6, 12, 18]
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(512, 256)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x_low, x_high):
        x_high = self.aspp(x_high)
        x_low = self.low_level_project(x_low)
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)
        x_concat = torch.cat([x_high_up, x_low], dim=1)
        return self.final_conv(x_concat)


class GhostResDeepLab(nn.Module):
    def __init__(self, num_classes=5, input_channels=12):
        super().__init__()
        self.encoder = ResNet34Encoder(input_channels=input_channels)
        self.decoder = DeepLabHead(num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        x_low, x_high = self.encoder(x)
        out = self.decoder(x_low, x_high)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out