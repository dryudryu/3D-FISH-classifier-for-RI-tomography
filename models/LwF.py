# LwF
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Densenet3d import DenseNet

def lwf169(**kwargs):
    model = LwF(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        last_features=1664,
        **kwargs)
    return model

class LwF(DenseNet):
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
        self.classifier_old = nn.Linear(last_features, num_classes)
        self.classifier_recon = nn.Linear(num_classes, last_features)

    def forward(self, x, reverse_alpha):
        feature = self.features(x)
        pool = self.avgpool(feature).view(x.size()[0],-1)

        out = self.classifier(pool)
        out_old = self.classifier_old(pool)
        recon = self.classifier_recon(out_old)
        return out, out_old, recon, pool

