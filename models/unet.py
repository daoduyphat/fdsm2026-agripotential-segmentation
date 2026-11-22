import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, output_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        d4 = self.drop4(c4)
        p4 = self.pool4(d4)

        c5 = self.conv5(p4)
        d5 = self.drop5(c5)

        u6 = self.up6(d5)
        m6 = torch.cat([d4, u6], dim=1)
        c6 = self.conv6(m6)

        u7 = self.up7(c6)
        m7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(m7)

        u8 = self.up8(c7)
        m8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(m8)

        u9 = self.up9(c8)
        m9 = torch.cat([c1, u9], dim=1)
        c9 = self.conv9(m9)

        return self.final(c9)
