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


def detect_defocus_region(args, image, device, transform):
    width, height = image.size
    ratio = height / width
    if width >= height:
        new_height = 224
        new_width = int((new_height / ratio) // 32 * 32)
    else:
        new_width = 224
        new_height = int((new_width * ratio) // 32 * 32)
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
    return detection


def main():
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='id of the gpu')
    parser.add_argument('--which-epoch', type=int, default=15, metavar='N',
                        help='which model to use')
    parser.add_argument('--input-path', type=str, default='defocus_images', help='test folder')
    parser.add_argument('--output-path', type=str, default='deblur_images', help='result folder')

    args = parser.parse_args()
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu_id if use_cuda else "cpu")

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    img_names = sorted(os.listdir(args.input_path))
    for img_name in img_names:
        image = Image.open(os.path.join(args.input_path, args.img_name))
        detection = detect_defocus_region(args, image, device, transform)
        tmp = img_name.split('.')
        tmp[-2] = tmp[-2] + '-dt'
        output_name = '.'.join(tmp)
        detection.save(os.path.join(args.output_path, output_name))


if __name__ == '__main__':
    main()
