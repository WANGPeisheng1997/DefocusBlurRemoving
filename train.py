from torch import nn
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
