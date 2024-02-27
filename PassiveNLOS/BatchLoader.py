# Coding :  UTF-8
# Author : Ziyu Zhan
import scipy.io
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class TrainDataset(Dataset):

    def __init__(self, path_dir,  transform):
        self.path_dir = path_dir
        self.transform = transform
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])
        self.image = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        image_item = self.image[item]
        image_path = os.path.join(self.path_dir, image_item)
        mat_vars = scipy.io.loadmat(image_path)
        image = mat_vars['G']
        image = Image.fromarray(image)
        input = mat_vars['label']
        input = Image.fromarray(input)
        if self.transform is not None:
            image = self.transform(image)
            input = self.transform_label(input)
        return image, input
