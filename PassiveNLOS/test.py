# 输入图像大小：400*400*3
# 模型：Down sampling (400->224) + ResNet
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import BatchLoader as bl
from torch.utils.data import DataLoader
from Unet_model import U_Net
from torchvision.utils import save_image
import os
import time



batch_size = 64
train_root = 'C:/Users/ziyuzhan/Desktop/SRT PNLOS/被动NLOS/NLOS/NLOS/Dataset/train_Unet'
test_root = 'C:/Users/ziyuzhan/Desktop/SRT PNLOS/被动NLOS/NLOS/NLOS/Dataset/test_Unet'

data_transforms = {
    'train':
    transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.01368), (0.00711))
    ]),
    'test':
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.01390), (0.00721))
    ]),
}
data_sets = {
    'train': bl.TrainDataset(train_root, transform=data_transforms['train']),
    'test': bl.TrainDataset(test_root, transform=data_transforms['test'])
}

dataloaders = {
    'train':
    DataLoader(data_sets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
    'test':
    DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)
}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

criterion = nn.L1Loss()

PATH = 'C:/Users/ziyuzhan/PycharmProjects/passiveNLOS/model_save/weights_0.h5'
model = U_Net()
model.load_state_dict(torch.load(PATH))
model = model.cuda() if use_cuda else model
print(model)

train_losses, test_losses = [], []
time_epoch = []

start = time.time()

running_loss = 0.0
running_corrects = 0

with torch.no_grad():
    model.eval()
    for inputs, labels in dataloaders['test']:
        if use_cuda:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        inputs = inputs.float()
        labels = labels.float()
        inputs = F.interpolate(inputs, 512)
        labels = F.interpolate(labels, 512)
        output = model(inputs)

        loss = criterion(output, labels)

        running_loss += loss.item()

# show_image = torch.cat([inputs, output, labels], dim=3)
# save_image(show_image, f"./show/test.png", nrow=2)


epoch_loss = running_loss / len(data_sets['test'])
epoch_acc = float(running_corrects) / len(data_sets['test'])

test_losses.append(epoch_loss)


print(f'test loss: {np.mean(test_losses):.4f}\n')
duration = time.time() - start
print("time: ", round(duration, 4))
time_epoch.append(duration)
print('end')

