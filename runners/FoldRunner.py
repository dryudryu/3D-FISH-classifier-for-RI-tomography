import torch
import os
from Logger import Logger
from .LymphoRunner import LymphoRunner

class FoldRunner(LymphoRunner):
    def train(self, loader_generator, test_loader):
        base_dir = self.save_dir
        for i, (train_loader, val_loader) in enumerate(loader_generator()):
            self.net.module.init_weight()
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.arg.lr, betas=self.arg.beta)
            self.save_dir = base_dir + "/fold%d"%(i)
            self.best_metric = -1 
            if os.path.exists(self.save_dir) is False:
                os.mkdir(self.save_dir)
            self.logger = Logger(self.save_dir)
            super().train(train_loader, val_loader, test_loader)
        self.save_dir = base_dir

    def test(self, loader_generator, test_loader):
        base_dir = self.save_dir
        test_acc = []
        for i, (train_loader, val_loader) in enumerate(loader_generator()):
            self.save_dir = base_dir + "/fold%d"%(i)
            self.logger = Logger(self.save_dir)
            super().load()
            test_acc.append(super().test(train_loader, val_loader, test_loader)[-1])
        self.save_dir = base_dir

        print("-------------")
        print("Fold %d Result"%(len(test_acc)))
        for i, acc in enumerate(test_acc):
            print("%d : %.4f"%(i, acc))
        print("Avg Test Acc : %.4f\n"%(sum(test_acc) / len(test_acc)))
        print("-------------")
