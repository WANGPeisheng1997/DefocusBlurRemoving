import argparse
import torch
from torch.utils.data import DataLoader
from dataset import DefocusDataset
from model import DetectionNet
from torch import optim
import train


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='id of the gpu')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--saving-interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving model')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu_id if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    path = "data"
    defocus_blur_dataset = DefocusDataset(path)
    defocus_blur_dataloader = DataLoader(defocus_blur_dataset, batch_size=8, shuffle=True, **kwargs)

    model = DetectionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train.train(args, model, device, defocus_blur_dataloader, optimizer, epoch)
        if epoch % args.saving_interval == 0:
            torch.save(model.state_dict(), "detect_%d.pt" % epoch)


if __name__ == '__main__':
    main()
