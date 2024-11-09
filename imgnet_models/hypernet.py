from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    #U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0, device = 'cuda'):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda(device)

    y = logits + gumbel_sample + offset
    return torch.sigmoid(y / T)

# def gumbel_softmax_sample(logits, T, offset=0):
#     if logits.is_cuda:
#         T = torch.Tensor([T]).cuda()
#     else:
#         T = torch.Tensor([T])
#     logits = logits + offset
#     RB = RelaxedBernoulli(T, logits=logits)
#     return RB.sample()

def hard_concrete(out, device):
    out_hard = torch.zeros(out.size())
    out_hard[out>=0.5]=1
    if out.is_cuda:
        out_hard = out_hard.cuda(device)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    out_hard = (out_hard - out).detach() + out
    return out_hard

def truncate_normal_(size, a=-1, b=1):
    values = truncnorm.rvs(a,b,size=size)
    values = torch.from_numpy(values).float()
    return values

class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #train = train
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        # print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda(grad_input.device)

        return grad_input * gw, None, None


class HyperStructure(nn.Module):
    '''
    Controller Network  incorporates bi-directional gated recurrent units (GRU) [9] fol- lowed by linear layers and Gumbel-Sigmoid [27] combined with straight-through estimator (STE)
    '''
    def __init__(self, structure=None, T=0.4, base=3, args=None):
        super(HyperStructure, self).__init__()
        self.device = args.gpu
        #self.fc1 = nn.Linear(64, 256, bias=False)
        self.bn1 = nn.LayerNorm([256]) # Layer

        self.T = T
        #self.inputs = 0.5*torch.ones(1,64)
        self.structure = structure # structure is the resnet structure
        # [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        
        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True) # input dim * output dim, input_size=10, hidden_size=20, num_layers=2

        self.h0 = torch.zeros(2,1,128) # bidirect * batch * output dim
        self.inputs = nn.Parameter(torch.Tensor(len(structure),1, 64)) # sequence len (structure len) * batch * input dim
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad=False
        print("HyperStructure init: structure", structure)

        # if wn_flag:
        #     self.linear_list = [weight_norm(Linear_GW(256, structure[i], bias=False, sparsity=self.sparsity[i])) for i in range(len(structure))]
        # else:
        self.linear_list = [nn.Linear(256, structure[i], bias=False) for i
                                in range(len(structure))]  # project to channel index in each layer

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        #decay
        self.base = base

        self.model_name = args.model_name
        if hasattr(args, 'block_string'):
            self.block_string = args.block_string
        if hasattr(args, 'se_list'):
            self.se_list = args.se_list

    def forward(self):
        if self.bn1.weight.is_cuda:
            pass
            self.inputs = self.inputs.cuda(self.device) # error
            self.h0 = self.h0.cuda(self.device)  # h0 has shape: 2,1,128  bidirect, batch, output dim
        outputs, hn = self.Bi_GRU(self.inputs, self.h0) # inputs shape: 30 = len(structure), 1, 64, h0 shape: 2,1,128  outputs shape: 30, 1, 128?
        outputs = [F.relu(self.bn1(outputs[i,:])) for i in  range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1) # sum channel size
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base, device=self.device) #  Gumbel-Sigmoid aims to produce a binary vector w, that approximates a binomial distribution
        if not self.training:
            out = hard_concrete(out, device=self.device) # =》 0 or 1
        # if self.training:
        #     self.update_bias()

        return out.squeeze()

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

    def resource_output(self):
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda(self.device)
            self.h0 = self.h0.cuda(self.device)
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)
        out = hard_concrete(out, self.device)

        return out.squeeze()

    def vector2mask(self, inputs):
        if self.model_name == 'resnet':
            if self.block_string == 'BasicBlock':
                return self.vector2mask_resnet(inputs)
            elif self.block_string == 'Bottleneck':
                return self.vector2mask_resnetbb(inputs)
        elif self.model_name == 'mobnetv2':
            return self.vector2mask_mobnetv2(inputs)
        elif self.model_name == 'mobnetv3':
            return self.vector2mask_mobnetv3(inputs)
        elif self.model_name == 'vae':
            print("vector2mask_vae called")
            return self.vector2mask_vae(inputs)
        
    def vector2mask_vae(self, inputs):
        vector = self.transfrom_output(inputs)
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            item_list.append(mask_output)
            item_list.append(mask_input)

            mask_list.append(item_list)
        return mask_list

    def vector2mask_resnet(self, inputs):
        vector = self.transfrom_output(inputs) # input => a serial of 0 / 1 value for each channel 
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_input)

            mask_list.append(item_list)
        return mask_list

    def vector2mask_resnetbb(self, inputs):
        vector = self.transfrom_output(inputs)
        mask_list = []
        length = len(vector)
        for i in range(0, length, 2):

        # for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask_middle_output = vector[i+1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i+1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            item_list.append(mask_output)
            item_list.append(mask_middle_input)
            item_list.append(mask_middle_output)
            item_list.append(mask_input)

            mask_list.append(item_list)
        return mask_list

    def vector2mask_mobnetv2(self, inputs):
        vector = self.transfrom_output(inputs)
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            item_list.append(mask_output)
            item_list.append(mask_middle)
            item_list.append(mask_input)

            mask_list.append(item_list)
        return mask_list

    def vector2mask_mobnetv3(self, inputs):
        vector = self.transfrom_output(inputs)
        mask_list = []
        se_list = self.se_list

        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            item_list.append(mask_output)
            item_list.append(mask_middle)
            item_list.append(mask_input)

            if se_list[i]:
                maskse_input = vector[i].unsqueeze(0)
                maskse_output = vector[i].unsqueeze(-1)

                item_list.append(maskse_input)
                item_list.append(maskse_output)

            mask_list.append(item_list)

        return mask_list


if __name__ == '__main__':
    net = HyperStructure()
    y = net()
    print(y)
