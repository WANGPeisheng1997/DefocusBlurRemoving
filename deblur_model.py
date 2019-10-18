import torch
from torch import nn


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation="relu", norm=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

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
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.res_1 = ResnetBlock(num_filter=in_channels * 2)
        self.res_2 = ResnetBlock(num_filter=in_channels * 2)
        self.res_3 = ResnetBlock(num_filter=in_channels * 2)
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_1(x)
        x = self.res_2(x)
        out = self.res_3(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpBlock, self).__init__()
        self.res_1 = ResnetBlock(num_filter=in_channels)
        self.res_2 = ResnetBlock(num_filter=in_channels)
        self.res_3 = ResnetBlock(num_filter=in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=(4, 4),
                                         stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        out = self.deconv(x)
        return out


class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.down_1 = DownBlock(32)
        self.down_2 = DownBlock(64)
        self.down_3 = DownBlock(128)
        self.res_1 = ResnetBlock(256)
        self.res_2 = ResnetBlock(256)
        self.res_3 = ResnetBlock(256)
        self.res_4 = ResnetBlock(256)
        self.res_5 = ResnetBlock(256)
        self.up_1 = UpBlock(256)
        self.up_2 = UpBlock(128)
        self.up_3 = UpBlock(64)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        c = self.conv_1(x)
        d1 = self.down_1(c)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        a1 = self.res_1(d3)
        a2 = self.res_2(a1)
        a3 = self.res_3(a2)
        a4 = self.res_4(a3)
        a5 = self.res_5(a4)
        u1 = self.up_1(a5)
        r1 = u1 + d2
        u2 = self.up_2(r1)
        r2 = u2 + d1
        u3 = self.up_3(r2)
        r3 = u3 + c
        out = self.conv_2(r3)
        return out

from PIL import Image
from torchvision import transforms
from torch import nn

if __name__ == '__main__':
    net = DeblurNet()
    print(net)
    # mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    features = net.forward(torch.randn(8, 3, 224, 224))
    print(features.size())  # 8 * 3 * 224 * 224
    # image1 = Image.open("test.png")
    # image2 = Image.open("data/HighResolutionImage/0002.png")
    # tensor1 = transforms.ToTensor()(image1)
    # tensor2 = transforms.ToTensor()(image2)
    # print(tensor1)
    # print(tensor2)
    # crit = nn.MSELoss()
    # print(crit(tensor1, tensor2))
