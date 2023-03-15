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

 "models.py" - Construction of arbitrary network topologies.

 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import function
from module import FA_wrapper, TrainingHook
import torchvision
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate, layer

encoder = encoding.PoissonEncoder()


class NetworkBuilder(nn.Module):
    """
    This version of the network builder assumes stride-2 pooling operations.
    """

    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, tau, train_mode, dropout,
                 conv_act, hidden_act, output_act, fc_zero_init, spike_window, device):
        super(NetworkBuilder, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window


        if (train_mode == "DFA") or (train_mode == "sDFA"):
            self.y = torch.zeros(train_batch_size, label_features, device=device)
            self.y.requires_grad = False
        else:
            self.y = None

        topology = topology.split('_')
        self.topology = topology
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV":
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(
                        output_dim / 2)  # /2 accounts for pooling operation of the previous convolutional layer
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(CNN_block(
                        in_channels=in_channels,
                        out_channels=int(layer[1]),
                        kernel_size=int(layer[2]),
                        stride=int(layer[3]),
                        padding=int(layer[4]),
                        bias=True,
                        activation=conv_act,
                        dim_hook=[label_features, out_channels, output_dim, output_dim],
                        label_features=label_features,
                        train_mode=train_mode
                    ))
                elif layer[0] == "FC":
                    if (i == 0):
                        input_dim = pow(input_size,2)*input_channels
                        # input_dim = input_size
                        self.conv_to_fc = 0
                        # print('i=0')
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])  # /2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                        # print('conv')
                    else:
                        input_dim = output_dim
                        # print('else')

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(FC_SNN_block(
                        in_features=input_dim,
                        out_features=output_dim,
                        bias=True,
                        tau = tau, 
                        activation=output_act if output_layer else hidden_act,
                        dropout=dropout,
                        dim_hook=None if output_layer else [label_features, output_dim],
                        label_features=label_features,
                        fc_zero_init=fc_zero_init,
                        train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode,
                        batch_size=self.batch_size,
                        spike_window=self.spike_window
                    ))

                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, input, labels):
        input = input.float().cuda()

        for step in range(self.spike_window):             
            x = encoder(input).float().cuda()

            for i in range(len(self.layers)):
                if i == self.conv_to_fc:
                    x = x.reshape(x.size(0), -1)
                x = self.layers[i](x, labels, self.y)

        x = self.layers[-1].sumspike / self.spike_window

        # print("x:",x.sum())
        # for i in range(len(smv_to_fc:
        #         x = x.reshape(x.size(0), -1)
        #     x = self.layers[i](x, labels, self.y)

        if x.requires_grad and (self.y is not None):
            self.y.data.copy_(x.data)  # in-place update, only happens with (s)DFA

        return x



class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, label_features, train_mode):
        super(CNN_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape, stride=stride, padding=padding)
        self.act = Activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.conv(x)
        x = self.act(x)
        x = self.hook(x, labels, y) #############3?
        x = self.pool(x)
        return x


class FC_SNN_block(nn.Module):
    def __init__(self, in_features, out_features, bias, tau, activation, dropout, dim_hook, label_features, fc_zero_init, train_mode, batch_size, spike_window):
        super(FC_SNN_block, self).__init__()
        self.spike_window = spike_window
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.lif = nn.Sequential(
            # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
            neuron.LIFNode(tau=tau)
            )  
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape) ##########FA 是否有问题？
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Sequential(
                neuron.LIFNode(tau=tau),
                nn.Dropout(self.dropout)
                )
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_features = out_features
        
    def forward(self, x, labels, y):
        # print("time",self.time_counter)
        
        if self.time_counter == 0:            
            self.sumspike = torch.zeros((self.batch_size, self.out_features)).cuda()

        self.time_counter += 1
        x = self.fc(x)    
        if self.dropout != 0:
            x = self.drop(x)
        else:
            x = self.lif(x)
        # x = self.act(x)
        # print(x)
        self.sumspike += x
        x = self.hook(x, labels, y)

        if self.time_counter == self.spike_window:
            self.time_counter = 0

        return x


class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)
