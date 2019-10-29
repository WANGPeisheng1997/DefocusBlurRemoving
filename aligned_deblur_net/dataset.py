from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import random
import cv2
import numpy
from scipy import interpolate


def get_disk_kernel(r):
    w = 2 * r + 1
    c = r
    kernel = numpy.zeros([w, w], numpy.float32)
    for x in range(r + 1):
        for y in range(r + 1):
            if x * x + y * y <= r * r:
                kernel[c + x, c + y] = 1.0
                kernel[c + x, c - y] = 1.0
                kernel[c - x, c + y] = 1.0
                kernel[c - x, c - y] = 1.0
    return kernel/numpy.sum(numpy.sum(kernel))


def gasuss_noise(img, mean=0, var=1):
    noise = numpy.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    # print(noise)
    out = numpy.clip(out, 0, 255)
    return out


def precompute_disk_kernel(lst):
    ret_dict = {}
    for r in lst:
        ret_dict[r] = get_disk_kernel(r)
    return ret_dict


kernel_dict = precompute_disk_kernel(list(range(1,24)))


def add_blur(image, mask):
    cv_img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    case = random.randint(1, 4)
    radius_1 = random.randint(2, 17)
    radius_2 = random.randint(3, 10)
    if case == 1:
        blur_img = cv2.filter2D(cv_img, -1, kernel_dict[radius_1])
    elif case == 2:
        dst_img_1 = cv2.filter2D(cv_img, -1, kernel_dict[radius_1])
        dst_img_2 = cv2.filter2D(cv_img, -1, kernel_dict[radius_2])
        blur_img = dst_img_1 * mask + dst_img_2 * (1 - mask)
    elif case == 3:
        dst_img_1 = cv2.filter2D(cv_img, -1, kernel_dict[radius_1])
        dst_img_2 = cv_img
        blur_img = dst_img_1 * mask + dst_img_2 * (1 - mask)
    else:
        sigma = random.uniform(0.5, 8.0)
        dst_img_1 = cv2.GaussianBlur(cv_img, (0, 0), sigma)
        sigma = random.uniform(0, 3.0)
        dst_img_2 = cv2.GaussianBlur(cv_img, (0, 0), sigma)
        blur_img = dst_img_1 * mask + dst_img_2 * (1 - mask)
    blur_img = gasuss_noise(blur_img)
    blur_img = blur_img.astype("uint8")
    pil_img = Image.fromarray(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
    return pil_img


def random_mask(h, w):
    if random.randint(1, 2) == 1:
        mask = _random_mask(w, h)
        mask = numpy.transpose(mask, [1, 0, 2])
    else:
        mask = _random_mask(h, w)
    if random.randint(1, 2) == 1:
        mask = 1 - mask
    return mask


def _random_mask(h, w):
    k = random.randint(3, 8)
    x = [0, h - 1]
    y = []
    for j in range(k - 2):
        new_x = random.randint(0, h - 1)
        while new_x in x:
            new_x = random.randint(0, h - 1)
        x.append(new_x)
    for j in range(k):
        y.append(random.randint(0, w - 1))
    x.sort()
    f = interpolate.interp1d(x, y, kind='quadratic')
    mask = numpy.zeros([h, w, 1])
    for x in range(h):
        y_mid = round(float(f(x)))
        for y in range(0, min(w, y_mid)):
            mask[x, y, 0] = 1
    return mask


class HighResolutionDataset(Dataset):

    def __init__(self, path):
        self._root = path
        self.image_files = [x.path for x in os.scandir(self._root) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        self._transform = transforms.Compose([transforms.Resize(1024),
                                              transforms.RandomCrop(512)
                                              ])

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        w, h = image.size
        if w > 1024 and h > 1024:
            gt_image = self._transform(image)
        else:
            gt_image = transforms.RandomCrop(512)(image)
        mask = random_mask(512, 512)
        blur_image = add_blur(gt_image, mask)
        # blur_image = transforms.ToTensor()(blur_image)
        # gt_image = transforms.ToTensor()(gt_image)
        return blur_image, gt_image

    def __len__(self):
        return len(self.image_files)
