import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownScaleBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(DownScaleBlock, self).__init__()
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv11 = nn.Conv2d(in_channels=in_channels * 2 + 3, out_channels=in_channels * 2, kernel_size=1)
        self.up_block = UpBlock(in_channels * 2)
        self.down_block = DownBlock(in_channels * 2)

    def forward(self, x, small):
        x = self.conv33(x)
        x = torch.cat((x, small), dim=1)
        x = self.conv11(x)
        x = self.up_block(x)
        x = self.down_block(x)
        return x


class UpScaleBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super(UpScaleBlock, self).__init__()
        self.deconv44 = nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=4,
                                           stride=2, padding=1)
        self.conv11 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1)
        self.up_block = UpBlock(out_channels)
        self.down_block = DownBlock(out_channels)

    def forward(self, x, hl_x):
        x = self.deconv44(x)
        x = torch.cat((x, hl_x), dim=1)
        x = self.conv11(x)
        x = self.up_block(x)
        x = self.down_block(x)
        return x


class DBRNet(nn.Module):
    def __init__(self):
        super(DBRNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.dsb_1 = DownScaleBlock(in_channels=32)
        self.dsb_2 = DownScaleBlock(in_channels=64)
        self.dsb_3 = DownScaleBlock(in_channels=128)
        self.usb_1 = UpScaleBlock(out_channels=128)
        self.usb_2 = UpScaleBlock(out_channels=64)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_8 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        h1 = self.conv_1(x)
        h2 = self.dsb_1(h1, x_2)
        h3 = self.dsb_2(h2, x_4)
        l3 = self.dsb_3(h3, x_8)
        l2 = self.usb_1(l3, h3)
        l1 = self.usb_2(l2, h2)
        d = self.deconv(l1)
        c = self.conv_2(torch.cat((d, h1), dim=1))
        out = self.conv_3(c)
        return out


from torch import nn

if __name__ == '__main__':
    net = DBRNet()
    input = torch.randn(1, 3, 256, 256)
    output = net(input)
    print(output.size())
