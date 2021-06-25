import os
import numpy as np

import torch
import torch.nn.functional as F

from Logger import Logger
from .LymphoRunner import LymphoRunner

from sklearn.metrics import confusion_matrix
from utils import get_confusion


class LwfRunner(LymphoRunner):
    def lwf_loss(self,
                out_n, out_o, out_recon, out_z,
                target, pseudo_label):
        loss_n = self.loss(out_n, target)
        loss_o = self.loss(out_o, pseudo_label)
        loss_r = F.mse_loss(out_recon, out_z)

        loss = loss_n + self.arg.lwf_alpha * loss_o + self.arg.lwf_beta * loss_r
        return loss

    def train(self, train_loader, val_loader, test_loader):
        # First Stage
        for i, (img, label) in enumerate(loader):
            img = img.to(self.torch_device)
            label = label.to(self.torch_device, dtype=torch.long)
            out, *_ = self.net(img)

            loss = self.loss(out, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        params_dict = dict(self.net.named_parameters())
        params_dict['module.classifier_old.bias'].data.copy_(params_dict['module.classifier.bias'].data)
        params_dict['module.classifier_old.weight'].data.copy_(params_dict['module.classifier.weight'].data)
            

        for epoch in range(self.start_epoch, self.epoch):
            self.net.train()
            for i, (img, label) in enumerate(loader):
                img = img.to(self.torch_device)
                label = label.to(self.torch_device, dtype=torch.long)
                out, out_o, recon, out_z = self.net(img)
                out_z = out_z.detach()

                with torch.no_grad():
                    self.net.eval()
                    _, out_o, *_ = self.net(input_)
                    _, pseudo_label = out_o.max(dim=1)
                    self.net.train()

                loss = self.lwf_loss(out, out_o, recon, out_z, label, pseudo_label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())

            self.valid(epoch, val_src_loader, val_tgt_loader, test_src_loader, test_tgt_loader)

    def _get_acc(self, loader, confusion=False):
        correct = 0
        preds, labels = [], []
        for i, (img, label) in enumerate(loader):
            img = img.to(self.torch_device)
            label = label.to(self.torch_device, dtype=torch.long)

            pred, *_ = self.net(src_img)
            _, idx = pred.max(dim=1)
            correct += torch.sum(label == idx).float().cpu().item()

            if confusion:
                preds  += idx.view(-1).tolist()
                labels += label.view(-1).tolist()
        
        if confusion:
            confusion = get_confusion(preds, labels)
        return correct / len(loader.dataset), confusion
