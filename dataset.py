from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms


class DefocusDataset(Dataset):

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
        # index 0 - 703
        # out_of_focus0001-0704
        img_name = "out_of_focus%04d" % (index + 1)
        img = Image.open(os.path.join(self._root, "images", img_name + ".jpg"))
        gt = Image.open(os.path.join(self._root, "gt", img_name + ".png"))
        img = self._transform_img(img)
        gt = self._transform_gt(gt)[0].long()
        return img, gt

    def __len__(self):
        return 703



