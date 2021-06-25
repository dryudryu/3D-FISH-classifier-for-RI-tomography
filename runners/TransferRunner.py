import torch

from glob import glob
import os
import time

from Logger import Logger
from .LymphoRunner import LymphoRunner
from .BaseRunner import BaseRunner

class TransferRunner(LymphoRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        # TODO : Call only baseRunner init method
        # super(BaseRunner, self).__init__(arg, torch_device, logger)
        self.arg = arg
        self.torch_device = torch_device 
        
        self.model_type = arg.model

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        
        self.save_dir = arg.save_dir

        self.logger = logger
        self.net = net
        self.loss = loss
        self.optim = optim

        self.best_metric = -1
        self.start_time = time.time()

        self.transfer_load()

    def transfer_load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                raise ValueError("Can't Loade file")
            filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File"%(self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            feature_dict = {k:v for k, v in ckpoint["network"].items() if "features" in k}
            model_dict = self.net.state_dict()
            model_dict.update(feature_dict)
            self.net.load_state_dict(model_dict)

            # TODO : Make loading
            self.optim = {
                "adam" : torch.optim.Adam(self.net.parameters(), lr=self.arg.lr, betas=self.arg.beta, weight_decay=self.arg.decay),
                "sgd"  : torch.optim.SGD(self.net.parameters(),
                                        lr=self.arg.lr, momentum=self.arg.momentum,
                                        weight_decay=self.arg.decay, nesterov=True)
            }[self.arg.optim]

            print("Transfer Load Model Type : %s, epoch : %d acc : %f"%(ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            raise ValueError("Can't Loade file")
