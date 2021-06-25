import os
import time

import torch
import torch.nn.functional as F

from Logger import Logger
from runners.LymphoRunner import LymphoRunner

from sklearn.metrics import confusion_matrix
from utils import get_confusion

from datas.Prefetcher import Prefetcher

class DropMaxRunner(LymphoRunner):

    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            self.net.train()
            for i, (input_, target_, path) in enumerate(train_loader):
                input_ = input_.to(self.torch_device)
                one_hot = torch.zeros(len(path), len(train_loader.dataset.classes))
                one_hot.scatter_(1, target_.unsqueeze(dim=1), 1)
                one_hot = one_hot.to(self.torch_device, non_blocking=True)

                o, p, r, q = self.net(input_)                
                loss = self.loss(o, p, r, q, one_hot)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader)
            else:
                self.save(epoch)


    def _get_acc(self, loader, confusion=False):
        correct = 0
        preds, labels = [], []
        if confusion:
            false_f = open(self.save_dir + "/false.txt", "w")

        K = len(loader.dataset.classes)
        for input_, target_, path in loader:
            input_ = input_.to(self.torch_device)
            target_ = target_.to(self.torch_device, non_blocking=True)

            o, p, r, q = self.net(input_)
            pred, idx = self.loss.get_acc(p, o, target_)
            correct += (idx == target_).sum().cpu().detach().item()

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

                idx_to_class = {v:k for k, v in loader.dataset.class_to_idx.items()}
                for i, (p, l) in enumerate(zip(idx.view(-1).tolist(), target_.view(-1).tolist())):
                    if p != l:
                        l = idx_to_class[l]
                        topk, indices = pred[i].topk(3)            
                        indices = [(idx_to_class[i], v) for i, v in zip(indices.view(-1).tolist(), topk.view(-1).tolist())]
                        title = "%s|Label : %s | Pred : %s : %.4f, %s : %.4f, %s : %.4f\n"%(path[i], l, *[z for i in indices for z in i])
                        false_f.write(title)

        if confusion:
            confusion = get_confusion(preds, labels) 
            false_f.close()
        return correct / len(loader.dataset), confusion
