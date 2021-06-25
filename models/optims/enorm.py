# https://github.com/facebookresearch/enorm/blob/master/enorm/enorm/enorm.py
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class ENorm():
    """
    Implements Equi-normalization for feedforward fully-connected and
    convolutional networks.
    Args:
        - named_params: the named parameters of your model, obtained as
          model.named_parameters()
        - optimizer: the optimizer, necessary for the momentum buffer update
          Note: only torch.optim.SGD supported for the moment
        - model_type: choose among ['linear', 'conv'] (see main.py)
        - c: asymmetric scaling factor that introduces a depth-wise penalty on
          the weights (default:1)
        - p: compute row and columns p-norms (default:2)
    Notes:
        - For all the supported architectures [fully connected, fully
          convolutional], we do not balance the last layer
        - In practice, we have found the training to be more stable when we do
          not balance the biases
    """

    def __init__(self, named_params, optimizer, model_type='resnet', c=1, p=2):
        self.named_params = list(named_params)
        self.optimizer = optimizer
        self.model_type = model_type
        self.momentum = self.optimizer.param_groups[0]['momentum']
        self.alpha = 0.5
        self.p = p

        # names to filter out
        # to_remove = ['bn']
        # fliter_map = lambda x: not any(name in x[0] for name in to_remove)
        
        # weights and biases
        self.weights = [(n, p) for n, p in self.named_params if 'weight' in n]
        # Note : For Remove BatchNorm
        # if Linear layer in models, not working
        self.weights = [(n, p) for n, p in self.weights if len(p.shape) > 1]
        self.biases = []
        self.n_layers = len(self.weights)

        # scaling vector
        self.n_layers = len(self.weights)
        self.C = [c] * self.n_layers
    
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d)

    def _get_weight(self, i, orientation='l'):
        _, param = self.weights[i]
        if self.model_type != 'linear':
            if orientation == 'l':
                # (C_in x k x k x k) x C_out
                param = param.view(param.size(0), -1).t()
            else:
                # C_in x (k x k x k x C_out)
                param = param.permute(1, 2, 3, 4, 0).contiguous().view(param.size(1), -1)
        return param

    def step(self):
        self.optimizer.step()
        self._step_conv()

    def _step_conv(self):
        print("w : ", self.weights[0][1].shape, self.weights[1][1].shape)
        left_w = self._get_weight(0, 'l')
        right_w = self._get_weight(1, 'r')
        print("StepConv w: ", left_w.shape, right_w.shape)
        left_norms = left_w.norm(p=2, dim=0).data
        right_norms = right_w.norm(p=2, dim=1).data
        print("StepConv norm: ", left_norms.shape, right_norms.shape)

        for i in range(1, self.n_layers - 1):
            balancer = (right_norms / (left_norms * self.C[i-1])).pow(self.alpha)

            left_w = self._get_weight(i, 'l')
            right_w = self._get_weight(i + 1, 'r')
            print(i, " w: ", self.weights[i][1].shape, self.weights[i+1][1].shape)
            print(i, " w: ", left_w.shape, right_w.shape)


            left_norms = left_w.norm(p=2, dim=0).data
            right_norms = right_w.norm(p=2, dim=1).data
            self.weights[i - 1][1].data.mul_(
                balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4))
            self.weights[i][1].data.mul_(
                1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(0))

            if self.momentum:
                self.optimizer.state[self.weights[i - 1][1]]['momentum_buffer'].mul_(
                    1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4))
                self.optimizer.state[self.weights[i][1]]['momentum_buffer'].mul_(
                    balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(0))
