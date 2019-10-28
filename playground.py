import os
import torch
import contextual_loss as cl
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

criterion = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
image1 = Image.open("0_gt.png").resize((32, 32))
image2 = Image.open("0_df.png").resize((32, 32))
tensor1 = transforms.ToTensor()(image1).unsqueeze(0)
tensor2 = transforms.ToTensor()(image2).unsqueeze(0)
loss = criterion(tensor1, tensor2)
print(loss)

# for i in range(1000):
#     os.makedirs("new_defocus_dataset/sharp_image/%d" % i)
#     os.makedirs("new_defocus_dataset/blut_image/%d" % i)