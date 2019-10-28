import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import HighResolutionDataset
from deblur_model import DeblurNet
from torch import optim
import time
import os


def train(args, model, device, defocus_blur_dataloader, optimizer, epoch):
    train_start_time = time.time()
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(defocus_blur_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        train_loss += float(loss.item()) * data.shape[0]
        loss.backward()
        optimizer.step()
    time_elapsed = time.time() - train_start_time
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / len(defocus_blur_dataloader.dataset)))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def load_network(args, network, device):
    save_path = os.path.join(args.saving_path, 'remove_%d.pt' % args.which_epoch)
    network.load_state_dict(torch.load(save_path, map_location=device))
    return network


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='id of the gpu')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--saving-interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving model')
    parser.add_argument('--saving-path', type=str, default='models', help='where to save the model')
    parser.add_argument('--which-epoch', type=int, default=0, help='which model to load')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    path = "data/HighResolutionImage"
    train_dataset = HighResolutionDataset(path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = DeblurNet().to(device)
    if args.which_epoch > 0:
        model = load_network(args, model, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.which_epoch + 1, args.which_epoch + args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch)
        if epoch % args.saving_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.saving_path, "remove_%d.pt" % epoch))


if __name__ == '__main__':
    main()
