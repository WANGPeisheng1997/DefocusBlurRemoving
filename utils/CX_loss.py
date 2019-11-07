import sys
sys.path.append('../')

import torch
import torch.nn as nn
from utils.VGG_model import VGG_Model
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

import time

class Distance_Type:
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2


class Contextual_Loss(nn.Module):
    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100, distance_type=Distance_Type.Cosine_Distance,
                 b=1.0, h=0.1, bilateral=True, feature_weight=0.05, device=None):
        super(Contextual_Loss, self).__init__()
        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        self.vgg_pred = VGG_Model(listen_list=listen_list)
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.h = h
        self.bilateral = bilateral
        self.feature_weight = feature_weight
        self.device = device
        self.pre_compute_L2 = None

    def forward(self, images, gt):

        if images.device.type == 'cpu':
            loss = torch.zeros(1)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone() for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
        else:
            id_cuda = torch.cuda.current_device()
            loss = torch.zeros(1).cuda(id_cuda)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}

        for key in self.layers_weights.keys():
            N, C, H, W = vgg_images[key].size()

            if self.crop_quarter:
                vgg_images[key] = self._crop_quarters()

            if H*W > self.max_1d_size**2:
                vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

            loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])
            loss += loss_t * self.layers_weights[key]

        # adp = nn.AdaptiveAvgPool2d((64, 64))
        # loss_rgb = self.calculate_CX_Loss(adp(images), adp(gt))
        # loss += loss_rgb

        return loss

    @staticmethod
    def _move_to_current_device(tensor):
        if tensor.device.type == 'cuda':
            id = torch.cuda.current_device()
            return tensor.cuda(id)
        return tensor

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = Contextual_Loss._move_to_current_device(indices)
        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature[..., round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB

            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _centered_by_T(I, T):
        mean_T = T.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        return I-mean_T, T-mean_T

    @staticmethod
    def _normalized_L2_channelwise(tensor):
        norms = tensor.norm(p=2, dim=1, keepdim=True)
        return tensor / norms

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        I_features, T_features = Contextual_Loss._centered_by_T(I_features, T_features)
        I_features = Contextual_Loss._normalized_L2_channelwise(I_features)
        T_features = Contextual_Loss._normalized_L2_channelwise(T_features)

        N, C, H, W = I_features.size()
        cosine_dist = []
        for i in range(N):
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous()
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist)
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)
        return cosine_dist

    def _compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0).to(self.device)

        return feature_grid

    @staticmethod
    def _compute_l2_distance(x, y):
        N, C, H, W = x.size()
        x_vec = x.view(N, C, -1)
        y_vec = y.view(N, C, -1)
        x_s = torch.sum(x_vec ** 2, dim=1)
        y_s = torch.sum(y_vec ** 2, dim=1)

        A = y_vec.transpose(1, 2) @ x_vec
        dist = y_s - 2 * A + x_s.transpose(0, 1)
        dist = dist.transpose(1, 2).reshape(N, H, W, H * W)
        dist = dist.clamp(min=0.)

        return dist

    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)
        return relative_dist

    def calculate_CX_Loss(self, I_features, T_features):

        calculate_start_time = time.time()

        I_features = Contextual_Loss._move_to_current_device(I_features)
        T_features = Contextual_Loss._move_to_current_device(T_features)

        if self.distanceType == Distance_Type.L1_Distance:
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == Distance_Type.L2_Distance:
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else:
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)

        if self.bilateral:
            b, c, h, w = T_features.size()
            if self.pre_compute_L2 is None:
                grid = self._compute_meshgrid((1, c, h, w))
                L2_distance = self._compute_l2_distance(grid, grid)
                self.pre_compute_L2 = L2_distance
            raw_distance = self.feature_weight * raw_distance + (1 - self.feature_weight) * self.pre_compute_L2

        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        del raw_distance

        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        del relative_distance

        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        del exp_distance

        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
        del contextual_sim

        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS))

        return CX_loss


if __name__ == '__main__':
    layers = {
            "conv_1_1": 1.0,
            "conv_3_2": 1.0
        }

    # image1 = Image.open("gt.png")
    # image2 = Image.open("1.png")
    # tensor1 = transforms.ToTensor()(image1).unsqueeze(0)
    # tensor2 = transforms.ToTensor()(image2).unsqueeze(0)
    tensor1 = torch.randn(1, 3, 32, 64)
    tensor2 = torch.randn(1, 3, 32, 64)

    contex_loss = Contextual_Loss(layers, max_1d_size=64)
    print(contex_loss(tensor1, tensor2))
