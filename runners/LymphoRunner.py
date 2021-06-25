import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import time
from utils import get_confusion


class LymphoRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)

        self.net = net
        self.loss = loss
        self.optim = optim
        self.arg = arg
        self.best_metric = -1
        self.start_time = time.time()

        self.load()

    def save(self, epoch, filename):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """

        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[self.arg.testfile])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" % (
            ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        self.net.train()
        for epoch in range(self.start_epoch, self.epoch):
            if epoch == 0:
                print("the first epoch starts") # flag 1
            for i, (input_, target_, path) in enumerate(train_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                loss = self.loss(output_, target_)


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                if i == len(train_loader)-1:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader)
            else:
                self.save(epoch)

    def _get_acc(self, loader, confusion=False):
        """ old version
        correct = 0
        preds, labels = [], []
        for input_, target_, _ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            correct += torch.sum(target_ == idx).float().cpu().item()

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

        if confusion:
            confusion = get_confusion(preds, labels)

        return correct / len(loader.dataset), confusion
        """
        correct = 0
        preds, labels = [], []
        featureArray = []
        gtArray = []
        predArray = []
        fDimArray = []
        if confusion:
            false_f = open(self.save_dir + "/false.txt", "w")

        for input_, target_, inputDir_ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, feature_, f1,f2,f3,f4,f5 = self.net(input_)
            

              
            _, idx = output_.max(dim=1)
            predArray.append(idx.cpu().detach().numpy())
            correct += torch.sum(target_ == idx).float().cpu().item()
            featureArray.append(feature_.cpu().detach().numpy())
            gtArray.append(target_.cpu().detach().numpy())

            #print(target_,idx, torch.sum(target_ == idx).float().cpu().item())
            
            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()
                idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}
                k = 3 if len(idx_to_class) > 3 else len(idx_to_class)
                for i, (p, l) in enumerate(zip(idx.view(-1).tolist(), target_.view(-1).tolist())):
                    if p != l:
                        l = idx_to_class[l]
                        topk, indices = output_[i].topk(k)
                        indices = [(idx_to_class[i], v) for i, v in
                                   zip(indices.view(-1).tolist(), topk.view(-1).tolist())]
                        title = "Label : %s | Pred : " % (l)
                        for pred_label, pred_value in indices:
                            title += "%s : %.4f," % (pred_label, pred_value)
                        title += "\n"
                        title += "(input data dir: %s " % (inputDir_[0])
                        title += "\n"
                        false_f.write(title)
         
        f1 = f1.cpu().detach().numpy()
        f2 = f2.cpu().detach().numpy()
        f3 = f3.cpu().detach().numpy()
        f4 = f4.cpu().detach().numpy()
        f5 = f5.cpu().detach().numpy()
         
        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()
        
        return correct / len(loader.dataset), confusion, featureArray, gtArray, predArray,f1,f2,f3,f4,f5

    def valid(self, epoch, val_loader, test_loader):
        # self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc(val_loader)
            test_acc, *_ = self._get_acc(test_loader)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)
            if acc > self.best_metric:# and epoch > 50:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))
            if test_acc > 0.95:
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))


    def _get_acc_25d(self, loader):
        patch_correct_sum = 0
        preds, labels = [], []

        cell_target = {}
        cell_correct = defaultdict(lambda: 0)
        for input_, target_, path in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            patch_correct = torch.sum(target_ == idx).float().cpu().item()
            patch_correct_sum += patch_correct

            for b in range(len(path)):
                cell_correct[path[b]] += output_[b]
                cell_target[path[b]] = target_[b]

        correct = 0
        for k, v in cell_correct.items():
            target_ = cell_target[k]
            _, idx = v.max(dim=0)
            correct += (target_ == idx).float().cpu().item()

            preds += idx.view(-1).tolist()
            labels += target_.view(-1).tolist()

        acc = correct / len(cell_correct.keys())

        idx_to_cls = {v: k for k, v in loader.dataset.class_to_idx.items()}
        preds = [idx_to_cls[i] for i in preds]
        labels = [idx_to_cls[i] for i in labels]
        a = confusion_matrix(labels, preds, labels=loader.dataset.classes)
        return acc, patch_correct_sum / len(loader.dataset), a
    

    def _test_25d(self, train_loader, val_loader, test_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, train_patch, _ = self._get_acc_25d(train_loader)
            valid_acc, valid_patch, _ = self._get_acc_25d(val_loader)
            test_acc, test_patch, test_confusion = self._get_acc_25d(test_loader)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(test_confusion)

    def test(self, train_loader, val_loader, test_loader):
        if self.arg.dim == "25d":
            return self._test_25d(train_loader, val_loader, test_loader)

        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, *_ = self._get_acc(train_loader)
            valid_acc, *_ = self._get_acc(val_loader)
            test_acc, test_confusion, fArray, gtArray, predArray,f1,f2,f3,f4,f5 = self._get_acc(test_loader, confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            
            np.save(self.save_dir + "/fArray.npy", fArray)
            np.save(self.save_dir + "/gtArray.npy", gtArray)
            np.save(self.save_dir + "/predArray.npy", predArray)
            #np.save(self.save_dir + "/f1.npy", f1)
            #np.save(self.save_dir + "/f2.npy", f2)
            #np.save(self.save_dir + "/f3.npy", f3)
            #np.save(self.save_dir + "/f4.npy", f4)
            #np.save(self.save_dir + "/f5.npy", f5)
            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(predArray)
            print(test_confusion)
           
        return train_acc, valid_acc, test_acc
