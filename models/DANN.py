# Domain Adversarial Training of Neural Netwrok
# DANN : https://arxiv.org/pdf/1505.07818.pdf
# http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
# http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.EffiDense3d import EffiDenseNet
from models.Densenet3d import DenseNet

from torch.autograd import Function
# TODO : Make Module
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def dann169(**kwargs):
    model = DANN(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model

class DANN(DenseNet):
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 norm="bn",
                 act="relu"):
        super().__init__(growth_rate=growth_rate,
                         block_config=block_config,
                         num_init_features=num_init_features,
                         bn_size=bn_size,
                         drop_rate=drop_rate,
                         num_classes=num_classes,
                         norm=norm, act=act)
        self.domain_classifier = nn.Linear(self.num_features, 1)

    def forward(self, x, reverse_alpha=0.25):
        feature = self.features(x)
        pool = self.avgpool(feature).view(x.size()[0],-1)
        reverse_pool = ReverseLayerF.apply(pool, reverse_alpha)

        out = self.classifier(pool)
        domain = self.domain_classifier(reverse_pool).view(-1)
        return out, domain

if __name__ == "__main__":
    net = dann169()
    a = torch.randn(1, 1, 120, 120, 70)
    net(a, 0.25)
    from torchsummary import summary
    summary(net.cuda(), (1, 120, 120, 70 ))   