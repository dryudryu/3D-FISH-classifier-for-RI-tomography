import os
import argparse
import torch
import torch.nn as nn
import time
# torch.backends.cudnn.benchmark = True

# example for mnist
from datas.LymphoLoader import lymphoLoader
from datas.CSVLoader import CSVLoader
from datas.FoldLoader import FoldGenerator

import utils
from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D

from Logger import Logger

from models.Densenet3d import d169_3d, d121_3d, d201_3d, dwdense_3d, d264_3d, dhy_3d
from models.EffiDense3d import ed169_3d
from models.fishnet import fishnet150, fishnet99, fishnetdw3
from models.fishnet import fishnetdw as fishdw_origin
from models.fishnet2 import fishdw2, fishdw, fish150
from models.fish_exfuse import fishdw_exfuse
from models.fish_dropmax import fishdw as fish_dropmax

from runners.LymphoRunner import LymphoRunner
from runners.FoldRunner import FoldRunner
from runners.TransferRunner import TransferRunner

"""parsing and configuration"""


def arg_parse():
    # projects description
    desc = "Lymphocyte Classifier"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")

    parser.add_argument('--dim', type=str, default='3d',
                        choices=["3d"])

    parser.add_argument('--aug', type=float, default=1, help='The number of Augmentation Rate')

    parser.add_argument('--norm', type=str, default='bn',
                        choices=["bn", "in"])

    parser.add_argument('--act', type=str, default='lrelu',
                        choices=["relu", "lrelu", "prelu"])

    parser.add_argument('--model', type=str, default='hydense',
                        choices=["attvgg", "dense169", "dense121", "dense201", "dwdense", "dense264", "hydense",
                                 "dpn92", "dpn98", "dpn107", "dpn131",
                                 "res18", "res34", "res50",
                                 "shake", "afd169", "sed169", "fish150", "fish99", "fishdworigin", "fishdw3origin",
                                 "fishdw", "fishdw2", "fishtest", "fishexfuse",],
                        help='The type of Models | vgg16 | dense | attvgg |')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    #parser.add_argument('--crop', type=int, default=160, help='The size of image cropping')
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--sampler', action="store_true", help='Weighted Sampler work')
    parser.add_argument('--testfile', type=int, default=-1, help='which test file selected')

    

    parser.add_argument('--optim', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=0.001)
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=1e-4)

    parser.add_argument('--fold', type=int, default=0, help='K-Fold Mode')
    parser.add_argument('--transfer', action="store_true", help='Weighted Sampler work')

    return parser.parse_args()


def get_model(arg, classes):
    if arg.model == "dense169":
        net = d169_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act)
    elif arg.model == "dense264":
        net = d264_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act)
    elif arg.model == "dense121":
        net = d121_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "hydense":
        net = dhy_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "dense201":
        net = d201_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "fish150":
        net = fish150(num_classes=classes)
    elif arg.model == "fish99":
        net = fishnet99(num_classes=classes)
    elif arg.model == "fishdw":
        net = fishdw(num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdw2":
        net = fishdw2(num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishexfuse":
        net = fishdw_exfuse(num_classes=classes)
    elif arg.model == "fishdworigin":
        net = fishdw_origin(num_classes=classes)
    elif arg.model == "fishdw3origin":
        net = fishnetdw3(num_classes=classes)
    return net


if __name__ == "__main__":
    arg = arg_parse()
    if arg.dim != "3d" and arg.model != "dense169":
        raise ValueError("Check dim, model")

    arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    torch.cuda.current_device()  # device flagship

    # CHANGE YOUR DATA DIRECTORY 
    data_path = "" 
    if arg.fold == 0:
        train_loader = lymphoLoader(data_path + "train", arg.batch_size, sampler=arg.sampler,
                                    transform=TRAIN_AUGS_3D, aug_rate=arg.aug,
                                    num_workers=arg.cpus, shuffle=True, drop_last=True)  
        val_loader = lymphoLoader(data_path + "val", arg.batch_size,
                                  transform=TEST_AUGS_3D, aug_rate=0,
                                  num_workers=arg.cpus, shuffle=True, drop_last=False)
        test_loader = lymphoLoader(data_path + "test", arg.batch_size,
                                   transform=TEST_AUGS_3D, aug_rate=0,
                                   num_workers=arg.cpus, shuffle=False, drop_last=False)
    else:
        loader_generator = FoldGenerator(data_path + "train", data_path + "val",
                                         arg.batch_size, arg.cpus, arg.aug, arg.fold)
        test_loader = lymphoLoader(data_path + "test", arg.batch_size,
                                   transform=TEST_AUGS_3D, aug_rate=0,
                                   num_workers=arg.cpus, shuffle=False, drop_last=False)
    print("dataset formulated")
    net = get_model(arg, classes=len(train_loader.dataset.classes))
    net = nn.DataParallel(net).to(torch_device)
    print("network formed")
    loss = nn.CrossEntropyLoss()

    optim = {
        "adam": torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "sgd": torch.optim.SGD(net.parameters(),
                               lr=arg.lr, momentum=arg.momentum,
                               weight_decay=arg.decay, nesterov=True)
    }[arg.optim]

    if arg.transfer:
        LymphoRunner = TransferRunner

    if arg.fold == 0:
        model = LymphoRunner(arg, net, optim, torch_device, loss,
                             logger)
        print("model defined")

        if arg.test is False:
            model.train(train_loader, val_loader, test_loader)

        start = time.time()
        model.test(train_loader, val_loader, test_loader)
        end = time.time()
        print(end - start)
        #model.net.eval()
        #with torch.no_grad():
            #acc, _ = model._get_acc(train_loader)
       # print(acc)
        #exit()