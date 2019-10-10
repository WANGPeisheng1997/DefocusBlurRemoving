from torch.utils.data import Dataset, DataLoader
import torch


class DefocusDataset(Dataset):

    def __init__(self, path):
        self.blur_image = torch.from_numpy(path)
        self.detection_map = torch.from_numpy(path)
        self.len = self.blur_image.shape[0]

    def __getitem__(self, index):
        return self.blur_image[index], self.detection_map[index]

    def __len__(self):
        return self.len


path = ""
defocus_blur_dataset = DefocusDataset(path)
defocus_blur_dataloader = DataLoader(defocus_blur_dataset, batch_size=16, num_workers=4, shuffle=True)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))