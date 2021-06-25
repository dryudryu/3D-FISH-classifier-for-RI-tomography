from datas.CSVLoader import CSVSet
from datas.CSVLoader import  _make_weighted_sampler

from datas.preprocess3d import TRAIN_AUGS_3D
from datas.preprocess3d import TEST_AUGS_3D

import random

from torch.utils import data



class FoldSet(CSVSet):
    def __init__(self, imgs, transform=None, aug_rate=0):
        self.origin_imgs = len(imgs)

        self.classes = sorted(list(set(i[1] for i in imgs)))
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.imgs = [(i, self.class_to_idx[t]) for i, t in imgs]

        
        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform

def FoldGenerator(train_csv, valid_csv, batch_size, cpus, aug_rate, fold):
    test_cnt = 30
    with open(train_csv, "r") as f:
        trains = [i.strip().split(";") for i in f.readlines()]
    with open(valid_csv, "r") as f:
        valids = [i.strip().split(";") for i in f.readlines()]

    imgs = trains + valids
    random.shuffle(imgs)

    classes = sorted(list(set(i[1] for i in imgs)))
    classes = {k:[] for k in classes}
    for path, class_ in imgs:
        classes[class_].append((path, class_))
    

    if len(imgs) < test_cnt * fold:
        raise IndexError("Image Length : %d, fold : %d"%(len(imgs, fold)))
    def _generator():
        for i in range(fold):
            train_imgs, valid_imgs = [], []
            for k, v in classes.items():
                valid_imgs += v[test_cnt * i:test_cnt * (i+1)]
                train_imgs += v[:test_cnt *  i] + v[test_cnt * (i+1):]
            train_set = FoldSet(train_imgs, transform=TRAIN_AUGS_3D, aug_rate=aug_rate)
            valid_set = FoldSet(valid_imgs, transform=TEST_AUGS_3D,  aug_rate=0)

            sampler = _make_weighted_sampler(train_set.imgs, len(classes.keys()))

            train_loader = data.DataLoader(train_set, batch_size, 
                                          sampler=sampler, num_workers=cpus, drop_last=True)
            valid_loader = data.DataLoader(valid_set, batch_size,
                                          num_workers=cpus, drop_last=False)                                                                            
            yield train_loader, valid_loader
    return _generator
