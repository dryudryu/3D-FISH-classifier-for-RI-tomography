import torch

import math

import torch.nn as nn

from models.layers.switcablenorm import SwitchNorm3d
from models.layers.fish_module import FishHead, FishBody, FishTail, Bridge

# https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py#L197
def _conv_bn_relu(in_ch, out_ch, stride=1, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                         norm(out_ch),
                         act())

class Fishnet(nn.Module):
    """
    Construct entire networks
    
    Args:
        start_c : Number of channels of input image
                  Note that it is NOT the number of channels in initial input image,
                            and it IS the number of output channel of stem
        num_cls : Number of classes
        stride : Stride of middle conv layer
        tail_num_blk : list of the numbers of Conv blocks in each FishTail stages
        body_num_blk : list of the numbers of Conv blocks in each FishBody stages
        head_num_blk : list of the numbers of Conv blocks in each FishHead stages
            (Note : `*_num_blk` includes 1 Residual blocks in the start of each stages)
        body_num_trans : list of the numbers of Conv blocks in transfer paths in each FishTail stages
        head_num_trans : list of the numbers of Conv blocks in transfer paths in each FishHead stages
        tail_channels : list of the number of in, out channel of each stages        
        body_channels : list of the number of in, out channel of each stages
        head_channels : list of the number of in, out channel of each stages

    """
    def __init__(self, start_c=64, num_cls=1000, norm="bn", act="relu",
                 tail_num_blk=[], bridge_num_blk=2,
                 body_num_blk=[], body_num_trans=[],
                 head_num_blk=[], head_num_trans=[],
                 tail_channels=[], body_channels=[], head_channels=[]):
        super().__init__()

        norm = {"bn":nn.BatchNorm3d,
                "in":nn.InstanceNorm3d, 
                "sn":SwitchNorm3d}[norm]

        act = {
            "relu":  lambda : nn.ReLU(inplace=True),
            "lrelu": lambda : nn.LeakyReLU(inplace=True),
            "prelu": lambda : nn.PReLU(),
        }[act]


        self.stem = nn.Sequential(
            _conv_bn_relu(1, start_c//2, stride=2, norm=norm, act=act),
            _conv_bn_relu(start_c//2, start_c//2, norm=norm, act=act),
            _conv_bn_relu(start_c//2, start_c, norm=norm, act=act),
            nn.MaxPool3d(3, padding=1, stride=2)
        )

        print("FishNet Initialzation Start")
        
        self.tail_layer = nn.ModuleList()
        for i, num_blk in enumerate(tail_num_blk):            
            layer = FishTail(tail_channels[i], tail_channels[i+1], num_blk, norm=norm, act=act)
            self.tail_layer.append(layer)

        self.bridge = Bridge(tail_channels[-1], bridge_num_blk, norm=norm, act=act)

        # First body module is not change feature map channel
        self.body_layer = nn.ModuleList()
        for i, (num_blk, num_trans) in enumerate(zip(body_num_blk, body_num_trans)):
            layer = FishBody(body_channels[i][0], body_channels[i][1], num_blk, 
                             tail_channels[-i-2], num_trans, dilation=2**i,
                             norm=norm, act=act)
            self.body_layer.append(layer)

        self.head_layer = nn.ModuleList()
        for i, (num_blk, num_trans) in enumerate(zip(head_num_blk, head_num_trans)):
            layer = FishHead(head_channels[i][0], head_channels[i][1], num_blk,
                             body_channels[-i-1][0], num_trans,
                             norm=norm, act=act)
            self.head_layer.append(layer)

        last_c = head_channels[-1][1]
        self.classifier = nn.Sequential(
            norm(last_c),
            act(),
            nn.Conv3d(last_c, last_c//2, 1, bias=False),
            norm(last_c//2),
            act(),
            nn.AdaptiveAvgPool3d(1),
        )

    
        self.o  = nn.Linear(last_c // 2, num_cls)
        self.ph = nn.Linear(last_c // 2, num_cls)
        self.rh = nn.Linear(last_c // 2, num_cls)

        self._init_weights()

    def forward(self, x):
        stem = self.stem(x)
        tail_features = [stem]
        for t in self.tail_layer:
            last_feature = tail_features[-1]
            tail_features.append(t(last_feature))

        bridge = self.bridge(tail_features[-1])

        body_features = [bridge]
        for b, tail in zip(self.body_layer, reversed(tail_features[:-1])):
            last_feature = body_features[-1]
            body_features.append(b(last_feature, tail))

        head_features = [body_features[-1]]
        for h, body in zip(self.head_layer, reversed(body_features[:-1])):
            last_feature = head_features[-1]            
            head_features.append(h(last_feature, body))

        x = self.classifier(head_features[-1]).view(x.shape[0], -1)
        f = head_features[-1] # extracted features for tsne 
        o = torch.sigmoid(self.o(x))
        ph, rh = self.ph(x), self.rh(x)

        p = torch.sigmoid(ph)
        r = torch.sigmoid(rh)
        # stop gradient on `ph`
        q = torch.sigmoid(ph.detach() + rh)
        return o,f,p,r,q

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# -- Factory -- 

def _calc_channel(start_c, num_blk):
    """
    Calculate the number of in and out channels of each stages in FishNet.

    Example:
        fish150 : start channel=64, num_blk=3,
        tail channels : Grow double in each stages,
                        [64, 128, 256 ...] = [start channel ** (2**num_blk) ....] 
        body channels : In first stage, in_channel and out_channel is the same,
                        but the other layers, the number of output channels is half of the number of input channel
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(512, 512), + 256
                         (768, 384), + 128
                         (512, 256)] + 64
        head channels : The number of input channels and output channels is the same.
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(320, 320),   + 512
                         (832, 832),   + 768
                         (1600, 1600)] + 512

    """

    tail_channels = [start_c]
    for i in range(num_blk):
        tail_channels.append(tail_channels[-1] * 2)
    print("Tail Channels : ", tail_channels)

    in_c, transfer_c = tail_channels[-1], tail_channels[-2]
    body_channels = [(in_c, in_c), (in_c + transfer_c, (in_c + transfer_c)//2)]
    # First body module is not change feature map channel
    for i in range(1, num_blk-1):
        transfer_c = tail_channels[-i-2]
        in_c = body_channels[-1][1] + transfer_c
        body_channels.append((in_c, in_c//2))
    print("Body Channels : ", body_channels)

    in_c = body_channels[-1][1] + tail_channels[0]
    head_channels = [(in_c, in_c)]
    for i in range(num_blk):
        transfer_c = body_channels[-i-1][0]
        in_c = head_channels[-1][1] + transfer_c
        head_channels.append((in_c, in_c))
    print("Head Channels : ", head_channels)
    return  {"tail_channels":tail_channels, "body_channels":body_channels, "head_channels":head_channels}
    
def fish99(num_classes=1000):
    start_c = 64

    tail_num_blk = [2, 2, 6]
    bridge_num_blk = 2

    body_num_blk = [1, 1, 1]
    body_num_trans = [1, 1, 1]

    head_num_blk = [1, 2, 2]
    head_num_trans = [1, 1, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_classes, 
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)

def fish150(num_classes=1000, norm="bn", act="relu"):
    start_c = 64

    tail_num_blk = [2, 4, 8]
    bridge_num_blk = 4

    body_num_blk = [2, 2, 2]
    body_num_trans = [2, 2, 2]

    head_num_blk = [2, 2, 4]
    head_num_trans = [2, 2, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_classes, norm, act,
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)

def fishdw(num_classes=1000, norm="bn", act="relu"):
    start_c = 64

    tail_num_blk = [2, 8, 16]
    bridge_num_blk = 8

    body_num_blk = [4, 4, 4]
    body_num_trans = [2, 2, 4]

    head_num_blk = [4, 4, 8]
    head_num_trans = [2, 2, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_classes, norm, act,
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)


def fishdw2(num_classes=1000, norm="bn", act="relu"):
    start_c = 64

    tail_num_blk = [4, 8, 16]
    bridge_num_blk = 8

    body_num_blk = [4, 4, 4]
    body_num_trans = [3, 3, 6]

    head_num_blk = [4, 4, 8]
    head_num_trans = [3, 3, 6]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_classes, norm, act,
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)
