import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, ratio=2, relu=True):
        super().__init__()
        init_channels = int(out_ch / ratio)
        new_channels = out_ch - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, kernel_size, stride, padding, bias=False),
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

class GhostConvBlockNested(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GhostConvBlockNested, self).__init__()
        self.conv = nn.Sequential(
            GhostConv(in_ch, out_ch),
            GhostConv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)

class PSAModule(nn.Module):
    def __init__(self, in_planes, planes, conv_kernels=None, stride=1, conv_groups=1):
        super(PSAModule, self).__init__()
        if conv_kernels is None: conv_kernels = [3, 5, 7, 9]
        self.conv_1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.sigmoid = nn.Sigmoid()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(planes // 4, planes // 4, kernel_size=k, padding=k // 2, groups=conv_groups, bias=False),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(inplace=True)
            ) for k in conv_kernels
        ])

    def forward(self, x):
        input_x = self.relu(self.bn_1(self.conv_1(x)))
        xs = torch.chunk(input_x, 4, dim=1)
        out_branches = [branch(xi) for branch, xi in zip(self.branches, xs)]
        x_concat = torch.cat(out_branches, dim=1)
        attention = self.sigmoid(self.bn_2(self.conv_2(x_concat)))
        return input_x * attention

class GRAUppAdvancedGhost(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, deep_supervision=True, pretrained=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, in_chans=input_channels, features_only=True, out_indices=(1, 2, 3, 4))
        enc = self.encoder.feature_info.channels()
        f = [64, 128, 256, 512]

        self.psa = PSAModule(enc[3], enc[3])
        self.conv2_1 = GhostConvBlockNested(enc[2] + enc[3], f[2])
        self.conv1_1 = GhostConvBlockNested(enc[1] + enc[2], f[1])
        self.conv1_2 = GhostConvBlockNested(enc[1] + f[1] + f[2], f[1])
        self.conv0_1 = GhostConvBlockNested(enc[0] + enc[1], f[0])
        self.conv0_2 = GhostConvBlockNested(enc[0] + f[0] + f[1], f[0])
        self.conv0_3 = GhostConvBlockNested(enc[0] + f[0] * 2 + f[1], f[0])

        self.final = nn.Conv2d(f[0], num_classes, 1)
        self.ds1 = nn.Conv2d(f[0], num_classes, 1)
        self.ds2 = nn.Conv2d(f[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        outs = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0 = outs[0], outs[1], outs[2], outs[3]
        x3_0 = self.psa(x3_0)

        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=False)], 1))

        out_final = F.interpolate(self.final(x0_3), size=input_size, mode='bilinear', align_corners=False)
        if self.deep_supervision and self.training:
            return [F.interpolate(self.ds1(x0_1), size=input_size, mode='bilinear', align_corners=False),
                    F.interpolate(self.ds2(x0_2), size=input_size, mode='bilinear', align_corners=False), out_final]
        return out_final