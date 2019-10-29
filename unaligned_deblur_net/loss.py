import torch
import numpy as np


class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        self.cs_weights_before_normalization = scaled_distances
        self.cs_NHWC = self.cs_weights_before_normalization

    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze(0)  # 1HWC --> 1CHW
            I_features_i = I_features[i, :, :, :].unsqueeze(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cs_flow.loss = (cs_flow.cosine_dist + 1) / 2
        cs_flow.loss = torch.clamp_min(cs_flow.loss, min=1e-5)
        cs_flow.loss = -torch.log(cs_flow.loss)
        return cs_flow.loss

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11


def CX_loss(T_features, I_features, alpha, device):
    def from_pt2tf(Tpt):
        Ttf = Tpt.permute(0, 2, 3, 1)
        return Ttf
    T_features_tf = from_pt2tf(T_features)
    I_features_tf = from_pt2tf(I_features)
    loss = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)

    position_tensor = torch.zeros(64, 64, 2).to(device)
    for i in range(64):
        for j in range(64):
            position_tensor[i][j][0] = i
            position_tensor[i][j][1] = j

    pos_vec = torch.reshape(position_tensor, (-1, 2))
    sq = torch.sum(pos_vec * pos_vec, 1)
    A = pos_vec @ torch.transpose(pos_vec, 0, 1)
    sq_c = sq.view(-1, 1)
    dist = sq_c - 2 * A + sq
    dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(64, 64, dist.shape[0]))
    dist = torch.clamp(dist, min=float(0.0))

    loss = loss + alpha * dist
    k_max_NC = torch.min(loss, dim=3)
    CS = torch.mean(torch.mean(k_max_NC[0], dim=1), dim=1)
    return CS, k_max_NC[1]


import torch.nn as nn

def cd_loss(input, target, alpha, device):
    SIZE = 64
    pooling = nn.AdaptiveAvgPool2d((SIZE, SIZE))
    input = pooling(input)
    target = pooling(target)
    loss, _ = CX_loss(input, target, alpha, device)
    loss = torch.mean(loss)
    return loss


import os
from PIL import Image, ImageDraw
from torchvision import transforms
import random

def get_CX_loss(input, target, alpha):
    gt_image = input
    target_image = target
    SIZE = 64
    gt_image = gt_image.resize((SIZE, SIZE))
    target_image = target_image.resize((SIZE, SIZE))
    t_t = transforms.ToTensor()
    t1 = t_t(gt_image).unsqueeze(0)
    t2 = t_t(target_image).unsqueeze(0)
    ls, corr = CX_loss(t1, t2, alpha)

    # concat = Image.new('RGB', (SIZE * 2, SIZE))
    # concat.paste(gt_image, (0, 0, SIZE, SIZE))
    # concat.paste(target_image, (SIZE, 0, SIZE * 2, SIZE))
    # corr = corr.squeeze()
    #
    # for i in range(SIZE):
    #     for j in range(SIZE):
    #         if random.randint(1, 400) == 1:
    #             pos = corr[i][j]
    #             row = pos // SIZE
    #             column = pos % SIZE
    #             draw = ImageDraw.Draw(concat)
    #             draw.line((j, i, SIZE + column, row), 'red')
    #
    # concat.show()
    return ls

import numpy as np
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % 3
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    gt = Image.open("gt.png")
    p1 = Image.open("1.png")
    p2 = Image.open("2.png")

    alphas = np.arange(0, 0.05, 0.001)
    for alpha in alphas:
        ls1 = get_CX_loss(gt, p1, alpha=alpha)
        ls2 = get_CX_loss(gt, p2, alpha=alpha)
        print("alpha:%f difference:%f, loss1:%f, loss%f" % (alpha, ls2-ls1, ls1, ls2))

