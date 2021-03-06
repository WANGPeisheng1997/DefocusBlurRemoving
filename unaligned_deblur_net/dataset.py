from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from utils.operations import random_crop, crop_to_multiple_of_k, random_mask, add_blur, random_resize, random_flip
import random
import sys


sys.path.append('../')


class FudanDefocusTrainDataset(Dataset):

    def __init__(self, path, size):
        self._root = path
        self._sharp_path = os.path.join(path, "sharp_images")
        self._blur_path = os.path.join(path, "blur_images")
        self._size = size

    def __getitem__(self, index):
        sharp_image_path = os.path.join(self._sharp_path, str(index))
        sharp_image_files = [x for x in os.scandir(sharp_image_path) if
                             x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        # get a random cropped sharp image:
        random.shuffle(sharp_image_files)
        sharp_random_image_name = sharp_image_files[0].name
        sharp_random_index = int(sharp_random_image_name.split("_")[1])
        sharp_random_image_path = sharp_image_files[0].path
        sharp_image = Image.open(sharp_random_image_path)
        # print(sharp_random_image_path)

        # all the crops
        blur_image_path = os.path.join(self._blur_path, str(index))
        # get the random index crop
        blur_image_path = os.path.join(blur_image_path, "%d_%d" % (index, sharp_random_index))
        blur_image_files = [x.path for x in os.scandir(blur_image_path) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        random.shuffle(blur_image_files)
        blur_image = Image.open(blur_image_files[0])

        # random_scale
        sharp_image, blur_image = random_resize(sharp_image, blur_image)

        # random flip
        sharp_image, blur_image = random_flip(sharp_image, blur_image)

        # random crop
        sharp_image, blur_image = random_crop(sharp_image, blur_image, 256)

        # to tensor
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)
        return blur_image, sharp_image

    def __len__(self):
        return self._size


class MixedDefocusTrainDataset(Dataset):

    def __init__(self, path, size):
        self._root = path
        self._sharp_path = os.path.join(path, "sharp_images")
        self._blur_path = os.path.join(path, "blur_images")
        self._size = size

        self._hr_root = "../data/HighResolutionImage"
        self._hr_image_files = [x.path for x in os.scandir(self._hr_root) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        self._transform = transforms.Compose([transforms.Resize(512),
                                              transforms.RandomCrop(256)
                                              ])

    def __getitem__(self, index):
        if index < self._size:
            sharp_image_path = os.path.join(self._sharp_path, str(index))
            sharp_image_files = [x for x in os.scandir(sharp_image_path) if
                                 x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
            # get a random cropped sharp image:
            random.shuffle(sharp_image_files)
            sharp_random_image_name = sharp_image_files[0].name
            sharp_random_index = int(sharp_random_image_name.split("_")[1])
            sharp_random_image_path = sharp_image_files[0].path
            sharp_image = Image.open(sharp_random_image_path)

            # all the crops
            blur_image_path = os.path.join(self._blur_path, str(index))
            # get the random index crop
            blur_image_path = os.path.join(blur_image_path, "%d_%d" % (index, sharp_random_index))
            blur_image_files = [x.path for x in os.scandir(blur_image_path) if
                                x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
            random.shuffle(blur_image_files)
            blur_image = Image.open(blur_image_files[0])
            sharp_image, blur_image = random_crop(sharp_image, blur_image, 256)

        else:
            image = Image.open(self._hr_image_files[index - self._size])
            w, h = image.size
            if w > 512 and h > 512:
                sharp_image = self._transform(image)
            else:
                sharp_image = transforms.RandomCrop(256)(image)
            mask = random_mask(256, 256)
            blur_image = add_blur(sharp_image, mask)

        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)

        return blur_image, sharp_image

    def __len__(self):
        return self._size + self._size // 2


class FudanDefocusTestDataset(Dataset):

    def __init__(self, path, size):
        self._root = path
        self._sharp_path = os.path.join(path, "sharp_images")
        self._blur_path = os.path.join(path, "blur_images")
        self._size = size

    def __getitem__(self, index):
        sharp_image_path = os.path.join(self._sharp_path, str(index))
        sharp_image_files = [x.path for x in os.scandir(sharp_image_path) if
                             x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        blur_image_path = os.path.join(self._blur_path, str(index))
        blur_image_files = [x.path for x in os.scandir(blur_image_path) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]

        sharp_image = Image.open(sharp_image_files[0])
        blur_image = Image.open(blur_image_files[0])
        # sharp_image = crop_to_multiple_of_k(sharp_image, 32)
        # blur_image = crop_to_multiple_of_k(blur_image, 32)
        sharp_image = sharp_image.crop((0, 0, 256, 256))
        blur_image = blur_image.crop((0, 0, 256, 256))
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)
        return blur_image, sharp_image

    def __len__(self):
        return self._size


if __name__ == '__main__':
    dataset = FudanDefocusTrainDataset("E:/GitHub/DefocusBlurRemoving/fudan_dataset/train", 128)
    blur, sharp = dataset[2]
    blur_image = transforms.ToPILImage()(blur)
    sharp_image = transforms.ToPILImage()(sharp)
    blur_image.show()
    sharp_image.show()
