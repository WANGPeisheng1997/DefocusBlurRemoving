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
    parser.add_argument('--which-epoch', type=int, default=10, metavar='N',
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
    data = transform(Image.open("test.jpg"))
    model_structure = DetectionNet().to(device)
    model = load_network(args, model_structure)
    model = model.to(device)
    model = model.eval()
    output = model(data)
    print(output)


if __name__ == '__main__':
    main()
