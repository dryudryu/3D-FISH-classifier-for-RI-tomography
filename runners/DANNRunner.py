import os
import numpy as np

import torch

from Logger import Logger
from .LymphoRunner import LymphoRunner

from sklearn.metrics import confusion_matrix
from utils import get_confusion


class DANNRunner(LymphoRunner):
    def train(self, train_loader, val_loader, target_loader):
        print("Src loader len :", len(train_loader), " Target loader len :", len(target_loader))
        source_domain_label = torch.zeros(self.arg.batch_size).to(self.torch_device, dtype=torch.float)
        target_domain_label = torch.ones(self.arg.batch_size).to(self.torch_device, dtype=torch.float)
        bce = torch.nn.BCEWithLogitsLoss()
        total_steps = self.epoch * len(train_loader)
        for epoch in range(self.start_epoch, self.epoch):
            self.net.train()
            start_steps = epoch * len(train_loader)
            # Source to class label, domain label
            # Target only domain label
            iter_target = iter(target_loader)
            for i, (source) in enumerate(train_loader):
                p = float(i + start_steps) / total_steps
                # default gamma = 10, theta = 1
                alpha = 2. / (1. + np.exp(-self.arg.dann_gamma * p)) - 1

                source_img = source[0].to(self.torch_device)
                source_label = source[1].to(self.torch_device)
                source_pred, source_domain = self.net(source_img, alpha)
                cls_loss = self.loss(source_pred, source_label)

                source_domain_loss = bce(source_domain, source_domain_label)

                if i % len(target_loader) == 0:
                    iter_target = iter(target_loader)
                    target = next(iter_target)
                else:
                    target = next(iter_target)

                target_img = target[0].to(self.torch_device)
                _, target_domain = self.net(target_img, alpha)
                target_domain_loss = bce(target_domain, target_domain_label)

                domain_loss = source_domain_loss + target_domain_loss

                loss = cls_loss + domain_loss * self.arg.dann_theta

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, dann_alpha=alpha,
                                          loss=loss.item(), cls_loss=cls_loss.item(),
                                          source_domain_loss=source_domain_loss.item(), target_domain_loss=target_domain_loss.item())

            self.valid(epoch, val_loader, target_loader)

    def _inference(self, loader, confusion=False, domain="src"):
        domain = 0 if domain == "src" else 1
        correct = 0; domain_correct = 0
        preds, labels = [], []
        for i, (input_, target_, _) in enumerate(loader):

            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, domain_pred = self.net(input_, 0)

            _, idx = output_.max(dim=1)
            correct += torch.sum(target_ == idx).float().cpu().item()
            domain_pred = torch.sigmoid(domain_pred)
            domain_pred[domain_pred > 0.5] = 1
            domain_pred[domain_pred <= 0.5] = 0
            domain_correct += (domain == domain_pred).cpu().sum().item()
            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()
        
        if confusion:
            confusion = get_confusion(preds, labels)            
        return correct / len(loader.dataset), confusion, domain_correct

    def valid(self, epoch, val_loader, test_loader):
        self.net.eval()
        with torch.no_grad():
            acc, _, domain_correct = self._inference(val_loader)
            test_acc, _, domain_target_correct = self._inference(test_loader, domain="target")
            domain_acc = (domain_correct + domain_target_correct) / (len(val_loader.dataset) + len(test_loader.dataset))
            self.logger.log_write("valid", epoch=epoch, acc=acc, domain_acc=domain_acc, test_acc=test_acc)

            if acc > self.best_metric:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]_domain[%.4f].pth.tar"%(epoch, acc, test_acc, domain_acc))


    def test(self, train_loader, val_loader, test_loader):
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, _, domain_train = self._inference(train_loader)
            valid_acc, _, domain_valid = self._inference(val_loader)
            test_acc, test_confusion, domain_test  = self._inference(test_loader, confusion=True, domain="target")

        domain_acc = (domain_train + domain_valid + domain_test) / (len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader))
        self.logger.log_write("test", fname="test",
                              train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc,
                              domain_acc=domain_acc)

        np.save(self.save_dir+"/test_confusion.npy", test_confusion)
        print(test_confusion)
        return train_acc, valid_acc, test_acc