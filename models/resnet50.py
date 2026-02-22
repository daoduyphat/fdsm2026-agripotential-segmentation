import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        if input_channels != 3:
            old = resnet.conv1
            new_conv1 = nn.Conv2d(input_channels, old.out_channels,
                                  kernel_size=old.kernel_size, stride=old.stride,
                                  padding=old.padding, bias=False)
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    new_conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv1

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

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
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(1, channels // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = x * self.sigmoid_channel(avg_out + max_out)

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return out * spatial_out

class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, attention=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.skip_bn = nn.BatchNorm2d(out_ch) if in_ch != out_ch else None

        self.attention = None
        if attention:
            if attention.lower() == 'cbam':
                self.attention = CBAM(out_ch)
            elif attention.lower() == 'eca':
                self.attention = ECA(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            x = self.skip_bn(self.skip(x))

        out = out + x
        if self.attention is not None:
            out = self.attention(out)

        return F.relu(out)

class ResNet50UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=5, input_channels=3, pretrained=True,
                 deep_supervision=True, attention='eca'):
        super().__init__()

        self.filters = [64, 256, 512, 1024, 2048]
        self.encoder = ResNet50Encoder(input_channels, pretrained)
        self.deep_supervision = deep_supervision

        self.reduce1 = nn.Conv2d(self.filters[1], self.filters[0], 1)
        self.reduce2 = nn.Conv2d(self.filters[2], self.filters[1], 1)
        self.reduce3 = nn.Conv2d(self.filters[3], self.filters[2], 1)
        self.reduce4 = nn.Conv2d(self.filters[4], self.filters[3], 1)

        self.conv0_1 = ResidualConv(self.filters[0] * 2, self.filters[0], attention)
        self.conv0_2 = ResidualConv(self.filters[0] * 3, self.filters[0], attention)
        self.conv0_3 = ResidualConv(self.filters[0] * 4, self.filters[0], attention)
        self.conv0_4 = ResidualConv(self.filters[0] * 5, self.filters[0], attention)

        self.conv1_1 = ResidualConv(self.filters[1] * 2, self.filters[1], attention)
        self.conv1_2 = ResidualConv(self.filters[1] * 3, self.filters[1], attention)
        self.conv1_3 = ResidualConv(self.filters[1] * 4, self.filters[1], attention)

        self.conv2_1 = ResidualConv(self.filters[2] * 2, self.filters[2], attention)
        self.conv2_2 = ResidualConv(self.filters[2] * 3, self.filters[2], attention)

        self.conv3_1 = ResidualConv(self.filters[3] * 2, self.filters[3], attention)

        self.final_x0_1 = nn.Conv2d(self.filters[0], num_classes, 1)
        self.final_x0_2 = nn.Conv2d(self.filters[0], num_classes, 1)
        self.final_x0_3 = nn.Conv2d(self.filters[0], num_classes, 1)
        self.final_x0_4 = nn.Conv2d(self.filters[0], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        s0 = (x0_0.shape[2], x0_0.shape[3])
        s1 = (x1_0.shape[2], x1_0.shape[3])
        s2 = (x2_0.shape[2], x2_0.shape[3])
        s3 = (x3_0.shape[2], x3_0.shape[3])

        x0_1 = self.conv0_1(
            torch.cat([x0_0, F.interpolate(self.reduce1(x1_0), s0, mode='bilinear', align_corners=True)], 1))
        x1_1 = self.conv1_1(
            torch.cat([x1_0, F.interpolate(self.reduce2(x2_0), s1, mode='bilinear', align_corners=True)], 1))
        x2_1 = self.conv2_1(
            torch.cat([x2_0, F.interpolate(self.reduce3(x3_0), s2, mode='bilinear', align_corners=True)], 1))
        x3_1 = self.conv3_1(
            torch.cat([x3_0, F.interpolate(self.reduce4(x4_0), s3, mode='bilinear', align_corners=True)], 1))

        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, F.interpolate(self.reduce1(x1_1), s0, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, F.interpolate(self.reduce2(x2_1), s1, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, F.interpolate(self.reduce3(x3_1), s2, mode='bilinear', align_corners=True)], 1))

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, F.interpolate(self.reduce1(x1_2), s0, mode='bilinear', align_corners=True)],
                      1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, F.interpolate(self.reduce2(x2_2), s1, mode='bilinear', align_corners=True)],
                      1))

        x0_4 = self.conv0_4(torch.cat(
            [x0_0, x0_1, x0_2, x0_3, F.interpolate(self.reduce1(x1_3), s0, mode='bilinear', align_corners=True)], 1))

        out_size = (x.shape[2], x.shape[3])
        out4 = F.interpolate(self.final_x0_4(x0_4), out_size, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            out1 = F.interpolate(self.final_x0_1(x0_1), out_size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.final_x0_2(x0_2), out_size, mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.final_x0_3(x0_3), out_size, mode='bilinear', align_corners=True)
            return [out1, out2, out3, out4]

        return out4