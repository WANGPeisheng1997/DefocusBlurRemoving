import argparse
import torch
from model import DetectionNet
import os
from torchvision import transforms
from PIL import Image


def load_network(args, network):
    save_path = os.path.join('detect_%d.pt' % args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def main():
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='id of the gpu')
    parser.add_argument('--which-epoch', type=int, default=15, metavar='N',
                        help='which model to use')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu_id if use_cuda else "cpu")

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    image = Image.open("test.jpg")
    width, height = image.size
    ratio = height / width
    new_height = 224
    new_width = int((new_height / ratio) // 32 * 32)
    image_resize = image.resize((new_width, new_height))
    data = transform(image_resize)
    data = data.unsqueeze(0).to(device)
    model_structure = DetectionNet().to(device)
    model = load_network(args, model_structure)
    model = model.to(device)
    model = model.eval()
    output = model(data).squeeze()
    result = torch.argmax(output, dim=0)
    result = result.unsqueeze(0).float()
    detection = transforms.ToPILImage()(result.cpu())
    detection = detection.resize(image.size)
    detection.save('detect.jpg')


if __name__ == '__main__':
    main()
