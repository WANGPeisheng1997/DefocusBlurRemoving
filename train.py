from torch import nn


def train(args, model, device, defocus_blur_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(defocus_blur_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(defocus_blur_dataloader.dataset),
                       100. * batch_idx / len(defocus_blur_dataloader), loss.item()))