from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
import torch
from PIL import Image
import os
from torchvision import transforms
from model import DetectionNet


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


path = "data"
defocus_blur_dataset = DefocusDataset(path)
defocus_blur_dataloader = DataLoader(defocus_blur_dataset, batch_size=8, num_workers=2, shuffle=True, pin_memory=True)

if __name__ == '__main__':
    model = DetectionNet()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for batch_idx, (data, target) in enumerate(defocus_blur_dataloader):
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(loss)



