from torch import nn
import torch


def test(args, model, device, defocus_blur_dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in defocus_blur_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss(reduction='sum')
            loss = criterion(output, target).item()
            test_loss += float(loss)

    test_loss /= len(defocus_blur_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
