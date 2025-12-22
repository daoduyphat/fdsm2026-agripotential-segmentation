import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------- Ghost Convolution --------------------
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


# -------------------- Ghost BasicBlock (ResNet block) --------------------
class GhostBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = GhostConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, relu=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GhostConv(planes, planes, kernel_size=3, stride=1, padding=1, relu=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# -------------------- ResNet34 Encoder --------------------
class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        if input_channels != 3:
            old = resnet.conv1
            new_conv1 = nn.Conv2d(input_channels, old.out_channels,
                                  kernel_size=old.kernel_size,
                                  stride=old.stride,
                                  padding=old.padding,
                                  bias=False)
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    new_conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv1

        def _make_layer(block, inplanes, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = [block(inplanes, planes, stride, downsample)]
            inplanes = planes
            for _ in range(1, blocks):
                layers.append(block(inplanes, planes))
            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool

        self.layer1 = _make_layer(GhostBasicBlock, 64, 64, 3)
        self.layer2 = _make_layer(GhostBasicBlock, 64, 128, 4, stride=2)
        self.layer3 = _make_layer(GhostBasicBlock, 128, 256, 6, stride=2)
        self.layer4 = _make_layer(GhostBasicBlock, 256, 512, 3, stride=2)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


# -------------------- Attention Modules --------------------

# ECA: Efficient Channel Attention
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


# CBAM Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        mid = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared(self.avg_pool(x))
        mx = self.shared(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


# CBAM Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg, mx], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


# CBAM Full Module
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# -------------------- Residual Conv with Attention --------------------
class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, attention=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.skip_bn = nn.BatchNorm2d(out_ch) if in_ch != out_ch else None

        att = attention.lower() if isinstance(attention, str) else None
        if att == 'cbam':
            self.attention = CBAM(out_ch)
        elif att == 'eca':
            self.attention = ECA(out_ch)
        else:
            self.attention = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            x = self.skip_bn(self.skip(x))

        out = out + x

        if self.attention is not None:
            out = self.attention(out)

        return F.relu(out)


# -------------------- Grad-CAM++ --------------------
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o)
        )
        self.target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0])
        )

    def generate(self, x, target_mask=None):
        self.model.zero_grad()
        y = self.model(x)
        y = y[-1] if isinstance(y, list) else y

        score = torch.sum(y * target_mask) if target_mask is not None else torch.sum(y)
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads ** 2, dim=(2, 3), keepdim=True)
        alpha = torch.sum(weights * acts, dim=(2, 3)) / \
                (torch.sum(weights * acts ** 2, dim=(2, 3)) * 2 + 1e-8)

        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        return cam.detach()


# -------------------- GRAU++ (Ghost-Res-Attention-UNet++) --------------------
class GRAUpp(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, pretrained=True,
                 deep_supervision=True, attention=None):

        super().__init__()
        f = [64, 64, 128, 256, 512]
        self.encoder = ResNet34Encoder(input_channels, pretrained)
        self.deep_supervision = deep_supervision

        att = attention.lower() if isinstance(attention, str) else None

        self.reduce1 = nn.Conv2d(f[1], f[0], 1)
        self.reduce2 = nn.Conv2d(f[2], f[1], 1)
        self.reduce3 = nn.Conv2d(f[3], f[2], 1)
        self.reduce4 = nn.Conv2d(f[4], f[3], 1)

        self.conv0_1 = ResidualConv(f[0] * 2, f[0], att)
        self.conv1_1 = ResidualConv(f[1] * 2, f[1], att)
        self.conv2_1 = ResidualConv(f[2] * 2, f[2], att)
        self.conv3_1 = ResidualConv(f[3] * 2, f[3], att)

        self.conv0_2 = ResidualConv(f[0] * 3, f[0], att)
        self.conv1_2 = ResidualConv(f[1] * 3, f[1], att)
        self.conv2_2 = ResidualConv(f[2] * 3, f[2], att)

        self.conv0_3 = ResidualConv(f[0] * 4, f[0], att)
        self.conv1_3 = ResidualConv(f[1] * 4, f[1], att)

        self.conv0_4 = ResidualConv(f[0] * 5, f[0], att)

        self.final_x0_1 = nn.Conv2d(f[0], num_classes, 1)
        self.final_x0_2 = nn.Conv2d(f[0], num_classes, 1)
        self.final_x0_3 = nn.Conv2d(f[0], num_classes, 1)
        self.final_x0_4 = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        x0_1 = self.conv0_1(torch.cat([x0_0,
                                       F.interpolate(self.reduce1(x1_0), x0_0.shape[2:], mode='bilinear')], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0,
                                       F.interpolate(self.reduce2(x2_0), x1_0.shape[2:], mode='bilinear')], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0,
                                       F.interpolate(self.reduce3(x3_0), x2_0.shape[2:], mode='bilinear')], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0,
                                       F.interpolate(self.reduce4(x4_0), x3_0.shape[2:], mode='bilinear')], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1,
                                       F.interpolate(self.reduce1(x1_1), x0_0.shape[2:], mode='bilinear')], 1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1,
                                       F.interpolate(self.reduce2(x2_1), x1_0.shape[2:], mode='bilinear')], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1,
                                       F.interpolate(self.reduce3(x3_1), x2_0.shape[2:], mode='bilinear')], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2,
                                       F.interpolate(self.reduce1(x1_2), x0_0.shape[2:], mode='bilinear')], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2,
                                       F.interpolate(self.reduce2(x2_2), x1_0.shape[2:], mode='bilinear')], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3,
                                       F.interpolate(self.reduce1(x1_3), x0_0.shape[2:], mode='bilinear')], 1))

        out1 = F.interpolate(self.final_x0_1(x0_1), size=x.shape[2:], mode='bilinear')
        out2 = F.interpolate(self.final_x0_2(x0_2), size=x.shape[2:], mode='bilinear')
        out3 = F.interpolate(self.final_x0_3(x0_3), size=x.shape[2:], mode='bilinear')
        out4 = F.interpolate(self.final_x0_4(x0_4), size=x.shape[2:], mode='bilinear')

        return [out1, out2, out3, out4] if self.deep_supervision else out4


# -------------------- Dice + BCE Loss --------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)
        inter = (logits * targets).sum()

        dice = (2 * inter + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice


bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()


def mixed_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)


# -------------------- Test --------------------
if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)

    model = GRAUpp(num_classes=1, input_channels=1, pretrained=False,
                   deep_supervision=True, attention=None)
    y = model(x)
    print("GRAU++ no attention:", [o.shape for o in y])

    model_cbam = GRAUpp(num_classes=1, input_channels=1, pretrained=False,
                        deep_supervision=True, attention='cbam')
    ycb = model_cbam(x)
    print("GRAU++ with CBAM:", [o.shape for o in ycb])

    model_eca = GRAUpp(num_classes=1, input_channels=1, pretrained=False,
                       deep_supervision=True, attention='eca')
    yec = model_eca(x)
    print("GRAU++ with ECA:", [o.shape for o in yec])
