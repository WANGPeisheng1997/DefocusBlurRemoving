from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import os
from torchvision import transforms
import random
import cv2
import numpy
from scipy import interpolate


def random_crop(img, gt, size):
    width, height = img.size
    max_x = width - size
    max_y = height - size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    region = (x, y, x + size, y + size)
    img = img.crop(region)
    gt = gt.crop(region)
    return img, gt


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


class DefocusTrainDataset(Dataset):

    def __init__(self, path):
        self._root = path
        self._transform_img = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
        self._transform_gt = transforms.Compose([transforms.ToTensor()
                                                 ])

    def __getitem__(self, index):
        # index 0 - 499
        # out_of_focus0001-0500
        img_name = "out_of_focus%04d" % (index + 1)
        img = Image.open(os.path.join(self._root, "train", "images", img_name + ".jpg"))
        gt = Image.open(os.path.join(self._root, "train", "gt", img_name + ".png"))
        img = transforms.Resize(256)(img)
        gt = transforms.Resize(256)(gt)
        img, gt = random_crop(img, gt, 224)
        img = self._transform_img(img)
        gt = self._transform_gt(gt)[0].long()
        return img, gt

    def __len__(self):
        return 500


class DefocusTestDataset(Dataset):

    def __init__(self, path):
        self._root = path
        self._transform_img = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
        self._transform_gt = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 ])

    def __getitem__(self, index):
        # index 0 - 203
        # out_of_focus0501-0704
        img_name = "out_of_focus%04d" % (index + 501)
        img = Image.open(os.path.join(self._root, "test", "images", img_name + ".jpg"))
        gt = Image.open(os.path.join(self._root, "test", "gt", img_name + ".png"))
        img = self._transform_img(img)
        gt = self._transform_gt(gt)[0].long()
        return img, gt

    def __len__(self):
        return 204


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


class RealDefocusDataset(Dataset):

    def __init__(self, path):
        self._root = path
        self._sharp_path = os.path.join(path, "sharp_image")
        self._blur_path = os.path.join(path, "blur_image")

    def __getitem__(self, index):
        sharp_image_path = os.path.join(self._sharp_path, str(index))
        sharp_image_files = [x.path for x in os.scandir(sharp_image_path) if
                             x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        blur_image_path = os.path.join(self._blur_path, str(index))
        blur_image_files = [x.path for x in os.scandir(blur_image_path) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]

        sharp_image = Image.open(sharp_image_files[0])
        random.shuffle(blur_image_files)
        blur_image = Image.open(blur_image_files[0])

        w, h = sharp_image.size
        if w > 1024 and h > 1024:
            sharp_image = transforms.Resize(1024)(sharp_image)
            blur_image = transforms.Resize(1024)(blur_image)
            sharp_image, blur_image = random_crop(sharp_image, blur_image, 512)
        else:
            sharp_image, blur_image = random_crop(sharp_image, blur_image, 512)

        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)
        return blur_image, sharp_image

    def __len__(self):
        return 61


if __name__ == '__main__':
    dataset = RealDefocusDataset("new_defocus_dataset")
    blur, gt = dataset[25]
    blur.show()
    gt.show()

    # image = Image.open("data/HighResolutionImage/0004.png")
    # image = transforms.Resize(1204)(image)
    # image = transforms.RandomCrop(512)(image)
    # width, height = image.size
    # mask = random_mask(height, width)
    # blur_image = add_blur(image, mask)
    # blur_image.save("test4.png")
