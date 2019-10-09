import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filter
    return torch.from_numpy(weight)


class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        backbone = vgg16.features
        self.conv_block_1 = backbone[:5]
        self.conv_block_2 = backbone[5:17]
        self.conv_block_3 = backbone[17:]
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_3 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv_1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(4, 4), stride=(2, 2),
                                           padding=(1, 1), bias=False)
        self.deconv_1.weight.data = bilinear_kernel(2, 2, 4)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(8, 8), stride=(4, 4),
                                           padding=(2, 2), bias=False)
        self.deconv_2.weight.data = bilinear_kernel(2, 2, 8)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(8, 8), stride=(4, 4),
                                           padding=(2, 2), bias=False)
        self.deconv_3.weight.data = bilinear_kernel(2, 2, 8)

    def forward(self, x):
        f1 = self.conv_block_1(x)
        f2 = self.conv_block_2(f1)
        f3 = self.conv_block_3(f2)
        c3 = self.conv_3(f3)
        d3 = self.deconv_3(c3)
        c2 = self.conv_2(f2)
        r2 = d3 + c2
        d2 = self.deconv_2(r2)
        c1 = self.conv_1(f1)
        r1 = d2 + c1
        d1 = self.deconv_1(r1)
        return d1


if __name__ == '__main__':
    net = DetectionNet()
    # mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    features = net.forward(torch.randn(8, 3, 224, 224))
    print(features)
    print(features.size())  # 8 * 2 * 224 * 224


'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''