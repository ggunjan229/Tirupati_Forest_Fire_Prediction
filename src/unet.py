"""Compact 2-D U-Net for demonstration patch masks."""
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=9, base=16):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = DoubleConv(in_channels, base), DoubleConv(base, base*2), DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base*8, base*4, 2, 2), DoubleConv(base*8, base*4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base*4, base*2, 2, 2), DoubleConv(base*4, base*2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base*2, base, 2, 2), DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        a = self.enc1(x); b = self.enc2(self.pool(a)); c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        z = self.dec3(torch.cat([self.up3(z), c], 1)); z = self.dec2(torch.cat([self.up2(z), b], 1))
        z = self.dec1(torch.cat([self.up1(z), a], 1))
        return self.out(z)
