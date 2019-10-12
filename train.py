import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import DefocusTrainDataset, DefocusTestDataset
from model import DetectionNet
from torch import optim
import time


def train(args, model, device, defocus_blur_dataloader, optimizer, epoch):
    train_start_time = time.time()
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(defocus_blur_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        train_loss += float(loss.item()) * data.shape[0]
        loss.backward()
        optimizer.step()
    time_elapsed = time.time() - train_start_time
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / len(defocus_blur_dataloader.dataset)))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def test(args, model, device, defocus_blur_dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in defocus_blur_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target).item()
            test_loss += float(loss) * data.shape[0]

    test_loss /= len(defocus_blur_dataloader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Defocus Blur Removing')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
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
    parser.add_argument('--saving-path', type=str, default='models', help='where to save the model')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu_id if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    path = "data"
    train_dataset = DefocusTrainDataset(path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = DefocusTestDataset(path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = DetectionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch)
        if epoch % args.saving_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.saving_path, "detect_%d.pt" % epoch))
        test(args, model, device, test_dataloader)


if __name__ == '__main__':
    main()
