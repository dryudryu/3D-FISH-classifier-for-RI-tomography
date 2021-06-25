# CCSA
# 
# [Official] https://github.com/samotiian/CCSA
# https://github.com/dainis-boumber/amamda/tree/37c45bed2accf221025a483f2368adc776632e9e/model/ccsa
# https://blog.lunit.io/2018/04/24/deep-supervised-domain-adaptation/
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Densenet3d import DenseNet

def ccsa169(**kwargs):
    model = CCSA(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        last_features=1664,
        **kwargs)
    return model

class CCSA(DenseNet):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 last_features,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 norm="bn",
                 act="relu"):
        super().__init__(sample_size,
                         sample_duration,
                         last_features,
                         growth_rate,
                         block_config,
                         num_init_features,
                         bn_size,
                         drop_rate,
                         num_classes,
                         norm,
                         act)

    def forward(self, x, reverse_alpha):
        feature = self.features(x)
        pool = self.avgpool(feature).view(x.size()[0],-1)

        out = self.classifier(pool)
        return out, feature

