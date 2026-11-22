import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv


class UNetPP(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, deep_supervision=False):
        super().__init__()

        self.deep_supervision = deep_supervision

        nb_filter = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder (dense connections)
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Upsampling blocks
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # Output layers
        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], output_channels, 1)
            self.final2 = nn.Conv2d(nb_filter[0], output_channels, 1)
            self.final3 = nn.Conv2d(nb_filter[0], output_channels, 1)
            self.final4 = nn.Conv2d(nb_filter[0], output_channels, 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], output_channels, 1)

    def forward(self, x):

        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # X0_1, X1_1, X2_1, X3_1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # X0_2, X1_2, X2_2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # X0_3, X1_3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # X0_4 final dense
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            return [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ]

        return self.final(x0_4)
