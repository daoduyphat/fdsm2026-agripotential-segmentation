import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None

        resnet = models.resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])

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

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(in_channels=2048, atrous_rates=[6, 12, 18], out_channels=256)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]

        x0 = self.conv1(x)
        x0 = self.maxpool(x0)

        low_level_features = self.layer1(x0)

        x_enc = self.layer2(low_level_features)
        x_enc = self.layer3(x_enc)
        x_enc = self.layer4(x_enc)

        x_aspp = self.aspp(x_enc)

        low_level_features = self.low_level_proj(low_level_features)

        x_aspp_up = F.interpolate(x_aspp, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat((x_aspp_up, low_level_features), dim=1)

        out = self.classifier(x_cat)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)

        return out