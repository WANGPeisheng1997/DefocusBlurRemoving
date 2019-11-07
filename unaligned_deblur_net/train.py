# CUDA_VISIBLE_DEVICES=X python train.py --w 0.05 --which-epoch 100
import sys
sys.path.append('../')

import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from unaligned_deblur_net.dataset import FudanDefocusTestDataset, FudanDefocusTrainDataset, MixedDefocusTrainDataset
from unaligned_deblur_net.model import DeblurNet
from torch import optim
import time
import os
from utils.CX_loss import Contextual_Loss
from torch.utils.tensorboard import SummaryWriter


def train(args, model, device, dataloader, optimizer, epoch, criterion):
    train_start_time = time.time()
    train_loss = 0
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += float(loss.item()) * data.shape[0]
        loss.backward()
        optimizer.step()
    training_time = time.time() - train_start_time
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / len(dataloader.dataset)))
    print('Training complete in {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
    return train_loss / len(dataloader.dataset)


def validate(args, model, device, dataloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            val_loss += float(loss) * data.shape[0]
    val_loss /= len(dataloader.dataset)
    print('Val Set Contextual Loss: {:.4f}'.format(val_loss))
    return val_loss


def load_network(args, network, device):
    save_path = os.path.join(args.saving_path, 'unaligned_%.2f_%d_RGB.pt' % (args.w, args.which_epoch))
    network.load_state_dict(torch.load(save_path, map_location=device))
    return network


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='id of the gpu')
    parser.add_argument('--saving-interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving model')
    parser.add_argument('--train-path', type=str, default='../fudan_dataset/train', help='where is the training set')
    parser.add_argument('--test-path', type=str, default='../fudan_dataset/test', help='where is the validating set')
    parser.add_argument('--saving-path', type=str, default='../models/unaligned', help='where to save the model')
    parser.add_argument('--which-epoch', type=int, default=0, help='which model to continue training')
    parser.add_argument('--w', type=float, default=0.05, help='feature weight')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.set_device(args.gpu_id)
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(torch.cuda.device_count())
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_path = args.train_path
    test_path = args.test_path
    train_dataset = FudanDefocusTrainDataset(train_path, 128)
    # train_dataset = MixedDefocusTrainDataset(train_path, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataset = FudanDefocusTestDataset(test_path, 8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, **kwargs)

    model = DeblurNet().to(device)
    if args.which_epoch > 0:
        model = load_network(args, model, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = Contextual_Loss({
        "conv_1_2": 1.0,
        "conv_2_2": 1.0,
        "conv_3_2": 1.0
    }, max_1d_size=64, feature_weight=args.w, device=device).to(device)

    writer = SummaryWriter()

    for epoch in range(args.which_epoch + 1, args.which_epoch + args.epochs + 1):
        train_loss = train(args, model, device, train_dataloader, optimizer, epoch, criterion)
        if epoch % args.saving_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.saving_path, 'unaligned_%.2f_%d_RGB.pt' % (args.w, epoch)))
        writer.add_scalar('Loss/train', train_loss, epoch)
        val_loss = validate(args, model, device, val_dataloader, criterion)
        writer.add_scalar('Loss/validate', val_loss, epoch)


if __name__ == '__main__':
    main()
