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
# from torch.utils.tensorboard import SummaryWriter
from Unet_model import U_Net
from torchvision.utils import save_image
import os
import time

# day = "unet"
# logs_dir = "logs_" + day
# writer = SummaryWriter(logs_dir)

save_dir = './model_save'
turn = 2
model_save_dir = "model_save/weights_" + str(turn) + ".h5"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

batch_size = 64
learning_rate = 0.001
num_epochs = 40
# train_root = 'C:/Users/ziyuzhan/Desktop/SRT PNLOS/被动NLOS/NLOS/NLOS/Dataset/train_Unet'
# test_root = 'C:/Users/ziyuzhan/Desktop/SRT PNLOS/被动NLOS/NLOS/NLOS/Dataset/test_Unet'
train_root = 'D:/清华/2023秋/NLOS_SRT/passiveNLOS/dataset/mat'
test_root = 'D:/清华/2023秋/NLOS_SRT/passiveNLOS/dataset/mat'

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
    DataLoader(data_sets['test'], batch_size=batch_size, shuffle=True, num_workers=0)   # shuffle
}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# model = models.resnet18(pretrained=True)
# model = model.cuda() if use_cuda else model
#
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(num_ftrs, 1024),
#     torch.nn.Dropout(0.2),
#     torch.nn.Linear(1024, 512),
#     torch.nn.Dropout(0.2),
#     torch.nn.Linear(512, 256),
#     torch.nn.Dropout(0.2),
#     torch.nn.Linear(256, 128),
#     torch.nn.Dropout(0.2),
#     torch.nn.Linear(128, 10),
#     torch.nn.Sigmoid()
# )
#
# model.fc = model.fc.cuda() if use_cuda else model.fc

model = U_Net()
model = model.cuda() if use_cuda else model

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# train_acces, test_acces = [], []
train_losses, test_losses = [], []
time_epoch = []
total_step = len(dataloaders['train'])
test_loss_min = np.Inf


for epoch in range(num_epochs):
    print(f'Epoch {epoch}\n')

    start = time.time()

    network_learned = False

    for phase in ['train', 'test']:

        running_loss = 0.0
        running_corrects = 0

        if phase == 'train':
            model.train()

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):

                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                inputs = inputs.float()
                labels = labels.float()
                inputs = F.interpolate(inputs,512)
                labels = F.interpolate(labels, 512)
                optimizer.zero_grad()
                output = model(inputs)
                # outputs = torch.clamp(output, max=1, min=-1)
                # labels = torch.squeeze(labels)
                # outputs = torch.softmax(outputs, dim=1)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # _, preds = torch.max(outputs, 1)
                # _, groudtruth = torch.max(labels, 1)
                running_loss += loss.item()
                # running_corrects += torch.sum(preds == groudtruth).item()
                if (batch_idx) % 20 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs - 1, batch_idx, total_step,
                                                                           loss.item()))
            # scheduler.step()
        else:
            with torch.no_grad():
                model.eval()
                for inputs, labels in dataloaders[phase]:
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
                    # outputs = torch.clamp(output, max=1, min=-1)
                    # labels = torch.squeeze(labels)
                    # outputs = torch.softmax(outputs, dim=1)

                    loss = criterion(output, labels)

                    # _, preds = torch.max(outputs, 1)
                    # _, groudtruth = torch.max(labels, 1)
                    running_loss += loss.item()
                    # running_corrects += torch.sum(preds == groudtruth).item()
            show_image = torch.cat([inputs, output, labels], dim=3)
            save_image(show_image, f"./show/test_{turn}_{epoch}.png", nrow=2)
            network_learned = running_loss < test_loss_min
            test_loss_min = running_loss if network_learned else test_loss_min

        epoch_loss = running_loss / len(data_sets[phase])
        epoch_acc = float(running_corrects) / len(data_sets[phase])

        if phase == 'train':
            # train_acces.append(epoch_acc * 100)
            train_losses.append(epoch_loss)
        else:
            # test_acces.append(epoch_acc * 100)
            test_losses.append(epoch_loss)

    # writer.add_scalar("train_loss", np.mean(train_losses), epoch)
    # writer.add_scalar("train_acc", np.mean(train_acces), epoch)
    # writer.add_scalar("test_loss", np.mean(test_losses), epoch)
    # writer.add_scalar("test_acc", np.mean(test_acces), epoch)

    # print(f'\ntrain-loss: {np.mean(train_losses):.4f}, train-acc: {train_acces[-1]:.4f}')
    # print(f'test loss: {np.mean(test_losses):.4f}, test acc: {test_acces[-1]:.4f}\n')
    print(f'\ntrain loss: {np.mean(train_losses):.4f}')
    print(f'test loss: {np.mean(test_losses):.4f}\n')
    duration = time.time() - start
    print("time: ", round(duration, 4))
    time_epoch.append(duration)

    if network_learned:
        torch.save(model.state_dict(), model_save_dir)
        print('Improvement-Detected, save-model')

    if epoch % 50 == 0 and epoch != 0:
        torch.save(model.state_dict(), "model_save/weights_" + str(turn) + '_' + str(epoch) + ".h5")
        print('Has passed 50 generations, save-model')

    with open('log_1.txt', 'w') as f:

        f.write('Hello, world!')
        f.write('\ntrain loss: ' + str(train_losses))
        f.write('\ntest loss: ' + str(test_losses))
        f.write('\ntime: ' + str(time_epoch))

# writer.close()

