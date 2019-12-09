import torch.utils.data as data
import torch
from PIL import Image

import os
import os.path
import sys
import numpy as np
import glob


class SimpleImageFolder(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filename_list = glob.glob(os.path.join(self.root, '*'))

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        img = Image.open(self.filename_list[idx])

        if self.transform:
            img = self.transform(img)

        return img


class CelebA(data.Dataset):
    def __init__(self, img_path, attr_path, transform=None, slice=[0, -1]):
        self.img_path = img_path
        self.attr_path = attr_path
        self.transform = transform
        self.slice = slice

        self.dataset = []
        self.idx2attr = {}
        self.attr2idx = {}
        self.read_attr()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, label = self.dataset[idx]

        img = Image.open(os.path.join(self.img_path, filename))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

    def read_attr(self):

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        classes = lines[1].split()

        for i in range(len(classes)):
            self.idx2attr[i] = classes[i]
            self.attr2idx[classes[i]] = i

        str_line = 2 + self.slice[0]
        end_line = 2 + self.slice[1] if self.slice[1] != -1 else -1

        for line in lines[str_line: end_line]:
            line = line.split()
            filename = line[0]
            label = []
            for i in line[1:]:
                if i == '1':
                    label.append(1)
                else:
                    label.append(0)

            self.dataset.append((filename, label))

    def get_classes(self):
        with open(self.attr_path, 'r') as f:
            lines = f.readlines()
        classes = lines[1].split()

        return classes


class CelebA_Slim(CelebA):
    def __init__(self, img_path, attr_path, transform=None, slice=[0, -1], attr=[15, 20, 31, 39]):
        self.attr = attr
        super(CelebA_Slim, self).__init__(img_path, attr_path, transform, slice)

    def read_attr(self):

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        classes = lines[1].split()
        classes = np.array(classes)
        classes = classes[self.attr]

        for i in range(len(classes)):
            self.idx2attr[i] = classes[i]
            self.attr2idx[classes[i]] = i

        str_line = 2 + self.slice[0]
        end_line = 2 + self.slice[1] if self.slice[1] != -1 else -1

        for line in lines[str_line: end_line]:
            line = line.split()
            filename = line[0]
            label = []
            for i in np.array(line[1:])[self.attr]:
                if i == '1':
                    label.append(1)
                else:
                    label.append(0)
                # print(label)

            self.dataset.append((filename, label))

    def get_classes(self):
        with open(self.attr_path, 'r') as f:
            lines = f.readlines()
        classes = lines[1].split()
        classes = np.array(classes)
        classes = classes[self.attr]

        return classes
