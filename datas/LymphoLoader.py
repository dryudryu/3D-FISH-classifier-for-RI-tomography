import os
import random

import numpy as np
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D
from datas.preprocess3d import mat2npy


def find_classes(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(path, class_to_idx):
    images = []
    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
    return images


class LymphoSet(data.Dataset):
    def __init__(self, dataset_path, transform=None, aug_rate=0):
        classes, class_to_idx = find_classes(dataset_path)
        print(class_to_idx)
        self.imgs = make_dataset(dataset_path, class_to_idx)

        self.origin_imgs = len(self.imgs)
        if len(self.imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + dataset_path))

        print("Dataset Dir : ", dataset_path, "len : ", len(self.imgs))

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

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

        for t in self.augs:
            img = t(img, ri=ri)

        """
        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)
        """
        return img, target, path

    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images, nclasses=6): 
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    print(count)
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


def lymphoLoader(image_path, batch_size, sampler=False,
                 transform=None, aug_rate=0,
                 num_workers=1, shuffle=False, drop_last=False):
    dataset = LymphoSet(image_path, transform=transform, aug_rate=aug_rate)
    if sampler:
        print("Sampler : ", image_path[-5:])
        sampler = _make_weighted_sampler(dataset.imgs)
        return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


if __name__ == "__main__":
    import torch

    # YOUR DATA DIRECTORY	
    data_path = ""
    import preprocess3d as preprocess

    pp = preprocess.TEST_AUGS_3D
    loader = lymphoLoader(data_path + "val", 3,
                          transform=pp, aug_rate=0,
                          num_workers=3, shuffle=False, drop_last=False)
    p1 = []
    for input, target, path in loader:
        p1 += list(path)

    p2 = []
    for input, target, path in loader:
        p2 += list(path)

    p3 = []
    for input, target, path in loader:
        p3 += list(path)

    print("3d aug!")
    # YOUR DATA DIRECTORY
    cc = len("/yourdata")+1

    for z1, z2, z3 in zip(p1, p2, p3):
        if z1 != z2 or z1 != z3 or z2 != z3:
            print(z1[cc:], z2[cc:], z3[cc:])

