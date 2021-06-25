import os
import random
from itertools import cycle

import numpy as np
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D
from datas.preprocess3d import mat2npy

"""
CCSA는 Target 비율이 극단적으로 적을 때 사용됬다.
Src : Tgt == 2000 : 10 ~ 70(sample_per_class)
"We are interested in the supervised domain adaptation 
when very few labeled target samples are available in training (from 1 to 7)." 라는데.....
우리는 거의 비율이 1:1(퉁쳐서) 인데 어떻게 사용할까?
"""
class CCSASet(data.Dataset):
    def __init__(self, path, transform=None, aug_rate=0, delim=";"):
        with open(path, "r") as f:
            paths = f.readlines()
        imgs =  [i.strip().split(delim) for i in path]

        self.classes = sorted(list(set(i[1] for i in imgs)))
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.imgs = [(i[0], self.class_to_idx[i[1]], i[2], self.class_to_idx[i[3]]) for i in imgs]

        self.origin_imgs = len(self.imgs)

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform

    def _get_img(self, path, index):
        mat = io.loadmat(path)
        img, ri = mat2npy(mat)
        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)
        return img

    def __getitem__(self, index):
        src_path, src_label, tgt_path, tgt_label = self.imgs[index]
        src_img = self._get_img(src_path, index)
        tgt_img = self._get_img(tgt_path, index)
        return src_img, tgt_img, src_label, tgt_label

    def __len__(self):
        return len(self.imgs)


def CCSALoader(csv_path, batch_size,
               sampler=False, transform=None, aug_rate=0,
               num_workers=1, shuffle=False, drop_last=False):
    dataset = CCSASet(csv_path, transform=transform, aug_rate=aug_rate)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
