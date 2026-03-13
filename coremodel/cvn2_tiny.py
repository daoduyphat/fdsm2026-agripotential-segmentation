import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------- Attention Modules --------------------

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
        avg_spatial = torch.mean(out, dim=1, keepdim=True)
        max_spatial, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        return out * spatial_out

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
        self.attention = None
        if attention:
            if attention.lower() == 'cbam':
                self.attention = CBAM(out_ch)
            elif attention.lower() == 'eca':
                self.attention = ECA(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.skip_bn(self.skip(x)) if self.skip is not None else x
        out = out + identity
        if self.attention is not None:
            out = self.attention(out)
        return F.relu(out)

# -------------------- ConvNeXt V2 Encoder --------------------

class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super().__init__()
        # Load ConvNeXt V2 Tiny
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', 
                                          pretrained=pretrained, 
                                          features_only=True)
        
        # Nếu input không phải 3 kênh
        if input_channels != 3:
            # Tìm lớp Conv2d đầu tiên (stem) một cách an toàn
            stem_layer = None
            parent_module = None
            attr_name = ""

            for name, module in self.backbone.named_modules():
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, nn.Conv2d):
                        stem_layer = sub_module
                        parent_module = module
                        attr_name = sub_name
                        break
                if stem_layer: break

            if stem_layer is None:
                raise AttributeError("Could not find the first Conv2d layer in the backbone.")

            # Tạo lớp mới
            new_stem = nn.Conv2d(input_channels, stem_layer.out_channels, 
                                 kernel_size=stem_layer.kernel_size, stride=stem_layer.stride, 
                                 padding=stem_layer.padding, bias=stem_layer.bias is not None)
            
            if pretrained:
                with torch.no_grad():
                    avg_weight = stem_layer.weight.mean(dim=1, keepdim=True)
                    new_stem.weight[:] = avg_weight.repeat(1, input_channels, 1, 1)
            
            # Thay thế lớp cũ
            setattr(parent_module, attr_name, new_stem)

    def forward(self, x):
        return self.backbone(x)

# -------------------- GRAU++ with ConvNeXt V2 Tiny --------------------

class GRAUpp_ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=5, input_channels=3, pretrained=True,
                 deep_supervision=True, attention='eca'):
        super().__init__()

        self.filters = [96, 192, 384, 768]
        self.encoder = ConvNeXtV2Encoder(input_channels, pretrained)
        self.deep_supervision = deep_supervision

        self.reduce1 = nn.Conv2d(self.filters[1], self.filters[0], 1)
        self.reduce2 = nn.Conv2d(self.filters[2], self.filters[1], 1)
        self.reduce3 = nn.Conv2d(self.filters[3], self.filters[2], 1)

        self.conv0_1 = ResidualConv(self.filters[0] * 2, self.filters[0], attention)
        self.conv1_1 = ResidualConv(self.filters[1] * 2, self.filters[1], attention)
        self.conv2_1 = ResidualConv(self.filters[2] * 2, self.filters[2], attention)
        self.conv0_2 = ResidualConv(self.filters[0] * 3, self.filters[0], attention)
        self.conv1_2 = ResidualConv(self.filters[1] * 3, self.filters[1], attention)
        self.conv0_3 = ResidualConv(self.filters[0] * 4, self.filters[0], attention)

        self.final_x0_1 = nn.Conv2d(self.filters[0], num_classes, 1)
        self.final_x0_2 = nn.Conv2d(self.filters[0], num_classes, 1)
        self.final_x0_3 = nn.Conv2d(self.filters[0], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0 = self.encoder(x)
        s0, s1, s2 = (x0_0.shape[2:], x1_0.shape[2:], x2_0.shape[2:])

        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(self.reduce1(x1_0), size=s0, mode='bilinear', align_corners=True)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(self.reduce2(x2_0), size=s1, mode='bilinear', align_corners=True)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(self.reduce3(x3_0), size=s2, mode='bilinear', align_corners=True)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(self.reduce1(x1_1), size=s0, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(self.reduce2(x2_1), size=s1, mode='bilinear', align_corners=True)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(self.reduce1(x1_2), size=s0, mode='bilinear', align_corners=True)], 1))

        out_size = x.shape[2:]
        out3 = F.interpolate(self.final_x0_3(x0_3), size=out_size, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            out1 = F.interpolate(self.final_x0_1(x0_1), size=out_size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.final_x0_2(x0_2), size=out_size, mode='bilinear', align_corners=True)
            return [out1, out2, out3]
        return out3