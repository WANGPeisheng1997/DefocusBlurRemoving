from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from utils.operations import random_crop, crop_to_multiple_of_k
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
        sharp_image_files = [x.path for x in os.scandir(sharp_image_path) if
                             x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        sharp_image_count = len(sharp_image_files)
        # get a random cropped sharp image:
        sharp_random_index = random.randint(0, sharp_image_count - 1)
        sharp_image = Image.open(sharp_image_files[sharp_random_index])
        print(sharp_image_files[sharp_random_index])

        # all the crops
        blur_image_path = os.path.join(self._blur_path, str(index))
        # get the random index crop
        blur_image_path = os.path.join(blur_image_path, "%d_%d" % (index, sharp_random_index))
        blur_image_files = [x.path for x in os.scandir(blur_image_path) if
                            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        random.shuffle(blur_image_files)
        blur_image = Image.open(blur_image_files[0])
        print(blur_image_files[0])

        # w, h = sharp_image.size
        # if w > 1024 and h > 1024:
        #     sharp_image = transforms.Resize(1024)(sharp_image)
        #     blur_image = transforms.Resize(1024)(blur_image)
        #     sharp_image, blur_image = random_crop(sharp_image, blur_image, 512)
        # else:
        #     sharp_image, blur_image = random_crop(sharp_image, blur_image, 512)
        sharp_image, blur_image = random_crop(sharp_image, blur_image, 256)

        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)
        return blur_image, sharp_image

    def __len__(self):
        return self._size


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
        sharp_image = crop_to_multiple_of_k(sharp_image, 32)
        blur_image = crop_to_multiple_of_k(blur_image, 32)
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image = transforms.ToTensor()(blur_image)
        return blur_image, sharp_image

    def __len__(self):
        return self._size