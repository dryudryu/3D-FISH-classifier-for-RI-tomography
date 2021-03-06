import os
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from cycler import cycler
import numpy as np
from scipy import io

# x axis of plot
LOG_KEYS = {
    "train": "epoch",
    "valid": "epoch",
    "test": "fname"
}

# y axis of plot
# save datas like loss, f1-score, PSNR, SSIM ..
# can multiple datas
LOG_VALUES = {
    "train": ["loss",
              #"cls_loss", "source_domain_loss", "target_domain_loss", "dann_alpha",  # DANN
              #"sa_loss", "sep_loss", "csa_loss"],
              ],
    "valid": ["acc", "test_acc", "domain_acc", "valid_acc",
              "tgt_acc", "test_src_acc", "test_tgt_acc"],
    "test": ["train_acc", "valid_acc", "test_acc", "domain_acc", "test_src_acc", "test_tgt_acc", "time"]
}


class Logger:

    def __init__(self, save_dir):
        self.log_file = save_dir + "/log.txt"
        self.buffers = []

    def will_write(self, line):
        print(line)
        self.buffers.append(line)

    def flush(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buffers))
            f.write("\n")
        self.buffers = []

    def write(self, line):
        self.will_write(line)
        self.flush()

    def log_write(self, learn_type, **values):
        """log write in buffers

        ex ) log_write("train", epoch=1, loss=0.3)

        Parmeters:
            learn_type : it must be train, valid or test
            values : values keys in LOG_VALUES
        """
 
        for k in values.keys():
            if k not in LOG_VALUES[learn_type] and k != LOG_KEYS[learn_type]:
                raise KeyError("%s Log %s keys not in log" % (learn_type, k))
        log = "[%s] %s" % (learn_type, json.dumps(values))
        self.will_write(log)
        if learn_type != "train":
            self.flush()

    def log_parse(self, log_key):
        log_dict = OrderedDict()
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) == 1 or not line.startswith("[%s]" % (log_key)):
                    continue
                # line : ~~
                line = line[line.find("] ") + 2:]  # ~~
                line_log = json.loads(line)

                train_log_key = line_log[LOG_KEYS[log_key]]
                line_log.pop(LOG_KEYS[log_key], None)
                log_dict[train_log_key] = line_log

        return log_dict

    def log_plot(self, log_key, mode="jupyter",
                 figsize=(12, 12), title="plot", colors=["C1", "C2"]):
        """Plotting Log graph

        If mode is jupyter then call plt.show.
        Or, mode is slack then save image and return save path

        Parameters:
            log_key : train, valid, test
            mode : jupyter or slack
            figsize : argument of plt
            title : plot title
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.legend(LOG_VALUES[log_key], loc="best")

        ax = plt.subplot(111)
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(LOG_VALUES[log_key])))
        ax.set_prop_cycle(cycler('color', colors))

        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        print(log_dict)
        print(x)
        for keys in LOG_VALUES[log_key]:
            y = [v[keys] for v in log_dict.values()]
            label = keys + ", max : %f" % (max(y))
            print(y)
            print(label)
            ax.plot(x, y, marker="o", linestyle="solid", label=label)
        ax.legend()

        if mode == "jupyter":
            plt.show()
        elif mode == "slack":
            # TODO : Test
            img_path = "tmp.jpg"
            plt.savefig(img_path)
            return img_path

    def plot_mat(self, log_key):
        """Plotting Log graph

        If mode is jupyter then call plt.show.
        Or, mode is slack then save image and return save path

        Parameters:
            log_key : train, valid, test
            mode : jupyter or slack
            figsize : argument of plt
            title : plot title
        """
        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        x2 = []
        for i in range(len(x)):
            x2.append(i)
        for keys in LOG_VALUES[log_key]:
            y = [v[keys] for v in log_dict.values()]
            y2 = np.array(y)
        x2 = np.array(x2)
        print(type(x2))
        print(type(y2))
        print(x2)
        print(y2)
        data1 = {'x': x2, 'y': y2}
        io.savemat('loss.mat',data1)



if __name__ == "__main__":
    logger = Logger("outs")
    logger.plot_mat()

