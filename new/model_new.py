import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ============================================================
# 1. PYRAMID SPLIT ATTENTION (PSA) MODULE
# ============================================================
class PSAModule(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=None, stride=1, conv_groups=1):
        super(PSAModule, self).__init__()
        if conv_kernels is None:
            conv_kernels = [3, 5, 7, 9]

        self.conv_1 = nn.Conv2d(inplans, planes, kernel_size=1, bias=False)
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
        input_x = self.conv_1(x)
        input_x = self.bn_1(input_x)
        input_x = self.relu(input_x)

        xs = torch.chunk(input_x, 4, dim=1)
        out_branches = [branch(xi) for branch, xi in zip(self.branches, xs)]
        x_concat = torch.cat(out_branches, dim=1)

        attention = self.conv_2(x_concat)
        attention = self.bn_2(attention)
        attention = self.sigmoid(attention)

        return x * attention


# ============================================================
# 2. NESTED CONV BLOCK (U-NET++ NODE)
# ============================================================
class ConvBlockNested(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlockNested, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 3. LOVASZ-SOFTMAX LOSS
# ============================================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).to(probas.dtype)
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return torch.tensor(0., requires_grad=True).to(probas.device)
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, probas, labels):
        b, c, h, w = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
        labels = labels.view(-1)

        valid = (labels != self.ignore_index)
        if valid.sum() == 0:
            return probas.sum() * 0.

        return lovasz_softmax_flat(probas[valid], labels[valid])


# ============================================================
# 4. GRAU++ ADVANCED ARCHITECTURE
# ============================================================
class GRAUppAdvanced(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, backbone_name='convnext_tiny', deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Backbone
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=input_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        enc = self.encoder.feature_info.channels()
        f = [64, 128, 256, 512]

        # Bottleneck Attention
        self.psa = PSAModule(enc[3], enc[3])

        # U-Net++ Nested Nodes
        self.conv2_1 = ConvBlockNested(enc[2] + enc[3], f[2])
        self.conv1_1 = ConvBlockNested(enc[1] + enc[2], f[1])
        self.conv1_2 = ConvBlockNested(enc[1] + f[1] + f[2], f[1])
        self.conv0_1 = ConvBlockNested(enc[0] + enc[1], f[0])
        self.conv0_2 = ConvBlockNested(enc[0] + f[0] + f[1], f[0])
        self.conv0_3 = ConvBlockNested(enc[0] + f[0] * 2 + f[1], f[0])

        # Heads
        self.final = nn.Conv2d(f[0], num_classes, 1)
        self.ds1 = nn.Conv2d(f[0], num_classes, 1)
        self.ds2 = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        x0_0, x1_0, x2_0, x3_0 = self.encoder(x)

        # Bottleneck
        x3_0 = self.psa(x3_0)

        # Decoder L2
        x2_1 = self.conv2_1(
            torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], 1))

        # Decoder L1
        x1_1 = self.conv1_1(
            torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], 1))

        # Decoder L0
        x0_1 = self.conv0_1(
            torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)],
                      1))

        # Final Output
        out_size = x.shape[2:]
        out_final = F.interpolate(self.final(x0_3), size=out_size, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            return [
                F.interpolate(self.ds1(x0_1), size=out_size, mode='bilinear', align_corners=True),
                F.interpolate(self.ds2(x0_2), size=out_size, mode='bilinear', align_corners=True),
                out_final
            ]

        return out_final
