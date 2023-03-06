# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "function.py" - Functional definition of the TrainingHook class (module.py).
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""


import torch
from torch.autograd import Function
from numpy import prod
import pdb
        

class HookFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, train_mode):
        if train_mode in ["DFA", "sDFA", "DRTP"]:
            ctx.save_for_backward(labels, y, fixed_fb_weights)
            # ctx.save_for_backward(input, labels, y, fixed_fb_weights)
        ctx.in1 = train_mode
        # torch.cuda.empty_cache()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        train_mode          = ctx.in1
        if train_mode == "BP":
            return grad_output, None, None, None, None
        elif train_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None
        
        # input, labels, y, fixed_fb_weights = ctx.saved_variables
        labels, y, fixed_fb_weights = ctx.saved_variables

        view_shape=grad_output.shape
        if train_mode == "DFA":
            # grad_output_est = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
            grad_output = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(view_shape)
            # print(f'dim y = {y.shape}\n dim labels={labels.shape} \n dim fixed_fb_weights={fixed_fb_weights.shape}')
        elif train_mode == "sDFA":
            grad_output = torch.sign(y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(view_shape)
        elif train_mode == "DRTP":
            grad_output = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(view_shape)
        else:
            raise NameError("=== ERROR: training mode " + str(train_mode) + " not supported")
        # pdb.set_trace()
        torch.cuda.empty_cache()
        # pdb.set_trace()
        return grad_output, None, None, None, None
        # return grad_output_est, None, None, None, None

trainingHook = HookFunction.apply
