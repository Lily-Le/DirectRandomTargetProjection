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

 "module.py" - Definition of hooks that allow performing FA, DFA, and DRTP training.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from function import trainingHook
import ipdb


class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim, stride=None, padding=None):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            # ipdb.set_trace()
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            # ipdb.set_trace()
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            # ipdb.set_trace()
            # torch.cuda.empty_cache()
            # ipdb.set_trace()
            return x
        else:
            # ipdb.set_trace()
            # torch.cuda.empty_cache()
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False
    
    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            # ipdb.set_trace()
            # torch.cuda.empty_cache()
            # ipdb.set_trace()
            if (self.layer_type == "fc"):
                # ipdb.set_trace()
                return self.output_grad.mm(self.fixed_fb_weights)
            elif (self.layer_type == "conv"):
                # ipdb.set_trace()
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad, self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            # ipdb.set_trace()
            return grad

    def FA_hook_post(self, grad):
        # ipdb.set_trace()
        # torch.cuda.empty_cache()
        self.output_grad = grad
        # ipdb.set_trace()
        return grad


class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, train_mode):
        super(TrainingHook, self).__init__()
        self.train_mode = train_mode
        assert train_mode in ["BP", "FA", "DFA", "DRTP", "sDFA", "shallow"], "=== ERROR: Unsupported hook training mode " + train_mode + "."
        
        # Feedback weights definition (FA feedback weights are handled in the FA_wrapper class)
        if self.train_mode in ["DFA", "DRTP", "sDFA"]:
        #dim hook for CNN [label_features,out_channels,output_dim,output_dim],
        #dim hook for FC [label_features,output_dim]
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        # print(torch.cuda.memory_summary())
        return trainingHook(input, labels, y, self.fixed_fb_weights, self.train_mode if (self.train_mode != "FA") else "BP") #FA is handled in FA_wrapper, not in TrainingHook

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.train_mode + ')'
