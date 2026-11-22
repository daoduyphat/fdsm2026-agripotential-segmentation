import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module (PPM) for UperNet."""

    def __init__(self, in_channels, pool_scales=(1, 2, 3, 6), out_channels=128):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        ppm_outs = [x] + [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class ConvNeXtV2UperNet(nn.Module):
    """ConvNeXt-V2 backbone with UperNet (PPM + FPN) decoder for 12-channel segmentation."""

    def __init__(self, input_channels=12, num_classes=5, backbone="convnextv2_atto", fpn_dim=128, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_channels,
        )
        enc_channels = self.encoder.feature_info.channels()

        self.ppm = PyramidPoolingModule(enc_channels[-1], out_channels=fpn_dim)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for c in enc_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in enc_channels[:-1]
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_dim * len(enc_channels), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.encoder(x)
        fpn_feats = [self.ppm(feats[-1])]

        for i in range(len(self.lateral_convs) - 1, -1, -1):
            lat = self.lateral_convs[i](feats[i])
            top_down = F.interpolate(fpn_feats[-1], size=lat.shape[2:], mode="bilinear", align_corners=False)
            fpn_feats.append(self.fpn_convs[i](lat + top_down))

        fpn_feats.reverse()
        target_size = fpn_feats[0].shape[2:]
        aligned = [fpn_feats[0]] + [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in fpn_feats[1:]
        ]
        logits = self.fusion(torch.cat(aligned, dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
