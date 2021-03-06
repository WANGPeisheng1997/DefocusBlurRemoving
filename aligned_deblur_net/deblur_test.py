import argparse
import torch
from deblur_model import DeblurNet
import os
from torchvision import transforms
from PIL import Image


def load_network(args, network, device):
    save_path = os.path.join(args.saving_path, 'remove_%d.pt' % args.which_epoch)
    network.load_state_dict(torch.load(save_path, map_location=device))
    return network


def deblur_defocus_image(args, image, device, transform):
    data = transform(image)
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        model_structure = DeblurNet().to(device)
        model = load_network(args, model_structure, device)
        model = model.to(device)
        model = model.eval()
        output = model(data).squeeze()
        output = output.clamp(0, 1)
        deblur = transforms.ToPILImage()(output.cpu())
    return deblur


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
    parser.add_argument('--saving-path', type=str, default='models', help='where is the model')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_id
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    img_names = sorted(os.listdir(args.input_path))
    for img_name in img_names:
        image = Image.open(os.path.join(args.input_path, img_name))
        width, height = image.size
        new_height = int(height // 32 * 32)
        new_width = int(width // 32 * 32)
        image_resize = image.resize((new_width, new_height))
        transform = transforms.Compose([transforms.ToTensor()])
        deblur = deblur_defocus_image(args, image_resize, device, transform)
        deblur = deblur.resize(image.size)
        tmp = img_name.split('.')
        tmp[-2] = tmp[-2] + '-rm'
        output_name = '.'.join(tmp)
        deblur.save(os.path.join(args.output_path, output_name))


if __name__ == '__main__':
    main()
