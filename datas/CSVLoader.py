import os
import random

import numpy as np
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D
from datas.preprocess3d import mat2npy


class CSVSet(data.Dataset):
    def __init__(self, csv_path, transform=None, aug_rate=0, delim=";"):
        with open(csv_path, "r") as f:
            imgs = f.readlines()
        imgs = [i.strip().split(delim) for i in imgs]
        
        self.classes = sorted(list(set(i[1] for i in imgs)))
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.imgs = [(i, self.class_to_idx[t]) for i, t in imgs]

        self.origin_imgs = len(self.imgs)
        if len(self.imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + csv_path))

        print("CSV Path : ", csv_path, "len : ", len(self.imgs))

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        mat = io.loadmat(path)
        img, ri = mat2npy(mat)
        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)
        return img, target, path


    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1

    N = float(sum(count))
    assert N == len(images)
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler


def CSVLoader(csv_path, batch_size, sampler=False,
              transform=None, aug_rate=0,
              num_workers=1, shuffle=False, drop_last=False):
    dataset = CSVSet(csv_path, transform=transform, aug_rate=aug_rate)
    if sampler:
        print("Sampler : ", csv_path[-5:])
        sampler = _make_weighted_sampler(dataset.imgs, nclasses=len(dataset.classes))
        return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
