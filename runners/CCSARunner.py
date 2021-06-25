import os
import numpy as np

import torch
import torch.nn.functional as F

from Logger import Logger
from .LymphoRunner import LymphoRunner

from sklearn.metrics import confusion_matrix
from utils import get_confusion


class CCSARunner(LymphoRunner):
    # Contrastive Semantic Alignment Loss
    # L_sa(g) + L_s(g)
    def csa_loss(self, src_feature, tgt_feature, domain):        
        dist = F.pairwise_distance(src_feature, tgt_feature, eps=1e-8)
        margin = 1.0
        loss_semantic_alignment = (1 - domain) * (margin - dist).clamp(min=0.0).pow(2)
        loss_seperation         = domain       * dist.pow(2)
        loss = torch.mean(loss_semantic_alignment + loss_seperation)
        return loss

    def train(self, train_loader, val_src_loader, val_tgt_loader, test_src_loader, test_tgt_loader):
        for epoch in range(self.start_epoch, self.epoch):
            self.net.train()
            # Source to class label, domain label
            # Target only domain label
            for i, (src_img, tgt_img, src_label, tgt_label) in enumerate(train_loader):
                src_img = src_img.to(self.torch_device)
                src_label = src_label.to(self.torch_device, dtype=torch.long)
                tgt_img = tgt_img.to(self.torch_device)
                tgt_label = tgt_label.to(self.torch_device, dtype=torch.long)

                src_pred, src_feature = self.net(src_img)
                cls_loss = self.loss(src_pred, src_label)

                _, tgt_feature = self.net(tgt_img)
                csa_loss = self.csa_loss(src_feature, tgt_feature, 
                                         (src_label == src.pred).float())

                loss = cls_loss * (1 - self.arg.ccsa_alpha) + csa_loss * self.arg.ccsa_alpha

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item(), 
                                          cls_loss=cls_loss.item(), csa_loss=csa_loss.item())

            self.valid(epoch, val_src_loader, val_tgt_loader, test_src_loader, test_tgt_loader)

    def _inference(self, loader, confusion=False):
        correct = 0
        preds, labels = [], []
        for i, (img, label) in enumerate(loader):
            img = img.to(self.torch_device)
            label = label.to(self.torch_device, dtype=torch.long)

            pred, _ = self.net(src_img)
            _, idx = pred.max(dim=1)
            correct += torch.sum(label == idx).float().cpu().item()

            if confusion:
                preds  += idx.view(-1).tolist()
                labels += label.view(-1).tolist()
        
        if confusion:
            confusion = get_confusion(preds, labels)
        return correct / len(loader.dataset), confusion

    def valid(self, epoch, val_src_loader, val_tgt_loader, test_src_loader, test_tgt_loader):
        self.net.eval()
        with torch.no_grad():
            acc, _ = self._inference(val_src_loader)
            tgt_acc, _ = self._inference(val_tgt_loader)
            test_src_acc, _ = self._inference(test_src_loader)
            test_tgt_acc, _ = self._inference(test_tgt_loader)

            self.logger.log_write("valid", epoch=epoch,
                                  acc=acc, tgt_acc=tgt_acc,
                                  test_src_acc=test_src_acc, test_tgt_acc=test_tgt_acc)

            if acc > self.best_metric and epoch > 20:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f].pth.tar"%(epoch, acc, test_acc))

    def test(self, train_src_loader, val_src_loader, test_src_loader,
            train_tgt_loader, val_tgt_loader, test_tgt_loader):
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_src_acc, _ = self._inference(train_src_loader)
            valid_src_acc, _ = self._inference(val_src_loader)
            test_src_acc, test_confusion  = self._inference(test_src_loader)

            train_tgt_acc, _ = self._inference(train_tgt_loader)
            valid_tgt_acc, _ = self._inference(val_tgt_loader)
            test_tgt_acc, test_confusion  = self._inference(test_tgt_loader)

        self.logger.log_write("test", fname="test",
                              train_acc=train_src_acc, valid_acc=valid_src_acc, test_acc=test_src_acc,
                              train_tgt_acc=train_tgt_acc, valid_tgt_acc=valid_tgt_acc, test_tgt_acc=test_tgt_acc,)

        np.save(self.save_dir+"/test_confusion.npy", test_confusion)
        print(test_confusion)
