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

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import function
from module import FA_wrapper, TrainingHook

from spikingjelly.clock_driven import neuron, encoding, functional, surrogate, layer

encoder = encoding.PoissonEncoder()


class NetworkBuilder(nn.Module):
    """
    This version of the network builder assumes stride-2 pooling operations.
    """
    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, train_mode, dropout, conv_act, hidden_act, output_act, fc_zero_init,loss, device,tau=2.0,spike_window=100,surrogate_function=surrogate.ATan()):
        super(NetworkBuilder, self).__init__()
        self.apply_softmax = (output_act == "none") and (loss == "CE")

        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window
        if (train_mode == "DFA") or (train_mode == "sDFA"):
            self.y = torch.zeros(train_batch_size, label_features, device=device)
            self.y.requires_grad = False
        else:
            self.y = None
        
        topology = topology.split('_')
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not all(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers-1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV":
                    in_channels  = input_channels if (i==0) else out_channels
                    out_channels = int(layer[1])
                    input_dim    = input_size if (i==0) else int(output_dim/2) #/2 accounts for pooling operation of the previous convolutional layer
                    output_dim   = int((input_dim - int(layer[2]) + 2*int(layer[4]))/int(layer[3]))+1
                    self.layers.append(CNN_block(
                                       in_channels=in_channels,
                                       out_channels=int(layer[1]),
                                       kernel_size=int(layer[2]),
                                       stride=int(layer[3]),
                                       padding=int(layer[4]),
                                       bias=True,
                                       activation=conv_act,
                                       dim_hook=[label_features,out_channels,output_dim,output_dim],
                                       label_features=label_features,
                                       train_mode=train_mode
                                       ))
                elif layer[0] == "CONV2":
                    in_channels  = input_channels if (i==0) else out_channels
                    out_channels = int(layer[1])
                    input_dim    = input_size if (i==0) else int(output_dim/2) #/2 accounts for pooling operation of the previous convolutional layer
                    output_dim   = int((input_dim - int(layer[2]) + 2*int(layer[4]))/int(layer[3]))+1
                    self.layers.append(CNN_block_VGG2(
                                       in_channels=in_channels,
                                       out_channels=int(layer[1]),
                                       kernel_size=int(layer[2]),
                                       stride=int(layer[3]),
                                       padding=int(layer[4]),
                                       bias=True,
                                       activation=conv_act,
                                       dim_hook=[label_features,out_channels,output_dim,output_dim],
                                       label_features=label_features,
                                       train_mode=train_mode
                                       ))                       
                elif layer[0] == "CONV3":
                    in_channels  = input_channels if (i==0) else out_channels
                    out_channels = int(layer[1])
                    input_dim    = input_size if (i==0) else int(output_dim/2) #/2 accounts for pooling operation of the previous convolutional layer
                    output_dim   = int((input_dim - int(layer[2]) + 2*int(layer[4]))/int(layer[3]))+1
                    self.layers.append(CNN_block_VGG3(
                                       in_channels=in_channels,
                                       out_channels=int(layer[1]),
                                       kernel_size=int(layer[2]),
                                       stride=int(layer[3]),
                                       padding=int(layer[4]),
                                       bias=True,
                                       activation=conv_act,
                                       dim_hook=[label_features,out_channels,output_dim,output_dim],
                                       label_features=label_features,
                                       train_mode=train_mode
                                       ))                         
                elif layer[0] == "CONVS": # Conv1 for SNN
                    in_channels  = input_channels if (i==0) else out_channels
                    out_channels = int(layer[1])
                    input_dim    = input_size if (i==0) else int(output_dim/2) #/2 accounts for pooling operation of the previous convolutional layer
                    output_dim   = int((input_dim - int(layer[2]) + 2*int(layer[4]))/int(layer[3]))+1
                    self.layers.append(Conv_SNN_block(
                                       in_channels=in_channels,
                                       out_channels=int(layer[1]),
                                       kernel_size=int(layer[2]),
                                       stride=int(layer[3]),
                                       padding=int(layer[4]),
                                       bias=True,
                                       tau=tau,
                                       surrogate_function=surrogate_function,
                                       dim_hook=[label_features,out_channels,output_dim,output_dim],
                                       label_features=label_features,
                                       train_mode=train_mode
                                       ))                                                                   
                elif layer[0] == "FC":
                    if (i==0):
                        input_dim = pow(input_size,2)*input_channels 
                        self.conv_to_fc = 0
                    elif topology_layers[i-1][0]=="CONV":
                        input_dim = pow(int(output_dim/2),2)*int(topology_layers[i-1][1]) #/2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim
                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers-1))
                    self.layers.append(FC_block(
                                       in_features=input_dim,
                                       out_features=output_dim,
                                       bias=True,
                                       activation=output_act if output_layer else hidden_act,
                                       dropout=dropout,
                                       dim_hook=None if output_layer else [label_features,output_dim],
                                       label_features=label_features,
                                       fc_zero_init=fc_zero_init,
                                       train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode
                                       ))

                elif layer[0] == "FCV": # FC for VGG
                    if (i==0):
                        input_dim = pow(input_size,2)*input_channels
                        self.conv_to_fc = 0
                    elif (topology_layers[i-1][0]=="CONV") or (topology_layers[i-1][0]=="CONV2") or (topology_layers[i-1][0]=="CONV3"):
                        input_dim = pow(int(output_dim/2),2)*int(topology_layers[i-1][1]) #/2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim
                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers-1))
                    self.layers.append(FC_block_VGG(
                                       in_features=input_dim,
                                       out_features=output_dim,
                                       bias=True,
                                       activation=output_act if output_layer else hidden_act,
                                       dropout=0 if output_layer else dropout, # 最后一层FC,不dropout
                                       dim_hook=None if output_layer else [label_features,output_dim],
                                       label_features=label_features,
                                       fc_zero_init=fc_zero_init,
                                       train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode
                                       ))       
                elif layer[0] == "FCS": # FC for SNN
                    if (i==0):
                        input_dim = pow(input_size,2)*input_channels 
                        self.conv_to_fc = 0
                    elif topology_layers[i-1][0]=="CONVS":
                        input_dim = pow(int(output_dim/2),2)*int(topology_layers[i-1][1]) #/2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim
                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers-1))
                    self.layers.append(FC_SNN_block(
                                       in_features=input_dim,
                                       out_features=output_dim,
                                       bias=True,
                                       tau=tau,
                                       dropout=dropout,
                                       dim_hook=None if output_layer else [label_features,output_dim],
                                       label_features=label_features,
                                       fc_zero_init=fc_zero_init,
                                       train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode
                                       ))                        
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, input, labels):
        # input = input.float().cuda()
        out_fr=0
        for step in range(self.spike_window):             
            # x = encoder(input).float().cuda()
            x=encoder(input)
            for i in range(len(self.layers)):
                if i == self.conv_to_fc:
                    x = x.reshape(x.size(0), -1)
                x = self.layers[i](x, labels, self.y)
            out_fr += x
        out_fr = out_fr/self.spike_window
           
        if out_fr.requires_grad and (self.y is not None):
            if self.apply_softmax:
                self.y.data.copy_(F.softmax(input=out_fr.data, dim=1)) # in-place update, only happens with (s)DFA
            else:
                self.y.data.copy_(out_fr.data) # in-place update, only happens with (s)DFA
        
        return out_fr


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init, train_mode):
        super(FC_block, self).__init__()
        
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        if self.dropout != 0:
            x = self.drop(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.hook(x, labels, y)
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
        x = self.conv(x) #batch*feature map size
        x = self.act(x) #labels: 0/1 label for each channel, y: probability 
        x = self.hook(x, labels, y) # y&label dim: batch*label_feature
        x = self.pool(x)
        return x

class FC_SNN_block(nn.Module):
    def __init__(self, in_features, out_features, bias, tau,  dropout, dim_hook, label_features, fc_zero_init, train_mode):
        super(FC_SNN_block, self).__init__()
        # self.spike_window = spike_window
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
        # self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Sequential(
                neuron.LIFNode(tau=tau),
                nn.Dropout(self.dropout)
                )
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        # self.sumspike = None
        # self.time_counter = 0
        # self.batch_size = batch_size
        self.out_features = out_features
        
    def forward(self, x, labels, y):
        # print("time",self.time_counter)
        
        # if self.time_counter == 0:            
            # self.sumspike = torch.zeros((self.batch_size, self.out_features)).cuda()

        # self.time_counter += 1
        x = self.fc(x)    
        if self.dropout != 0:
            x = self.drop(x)
        else:
            x = self.lif(x)
        # x = self.act(x)
        # print(x)
        # self.sumspike += x
        x = self.hook(x, labels, y)

        # if self.time_counter == self.spike_window:
        #     self.time_counter = 0

        return x
   
class Conv_SNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, tau,surrogate_function, dim_hook, label_features, train_mode):
        super(Conv_SNN_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape, stride=stride, padding=padding)
        self.act = nn.Sequential(neuron.LIFNode(tau=tau,surrogate_function=surrogate_function))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.conv(x) #batch*feature map size
        x = self.act(x) #labels: 0/1 label for each channel, y: probability 
        x = self.hook(x, labels, y) # y&label dim: batch*label_feature
        x = self.pool(x)
        return x
    


class CNN_block_VGG2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, label_features, train_mode):
        super(CNN_block_VGG2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv1 = FA_wrapper(module=self.conv1, layer_type='conv', dim=self.conv1.weight.shape, stride=stride, padding=padding)
        self.act1 = Activation(activation)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv2 = FA_wrapper(module=self.conv2, layer_type='conv', dim=self.conv2.weight.shape, stride=stride, padding=padding)
        self.act2 = Activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.hook(x, labels, y)
        x = self.pool(x)
        return x   

class CNN_block_VGG3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, label_features, train_mode):
        super(CNN_block_VGG3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv1 = FA_wrapper(module=self.conv1, layer_type='conv', dim=self.conv1.weight.shape, stride=stride, padding=padding)
        self.act1 = Activation(activation)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv2 = FA_wrapper(module=self.conv2, layer_type='conv', dim=self.conv2.weight.shape, stride=stride, padding=padding)
        self.act2 = Activation(activation)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv3 = FA_wrapper(module=self.conv3, layer_type='conv', dim=self.conv3.weight.shape, stride=stride, padding=padding)
        self.act3 = Activation(activation)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)

        x = self.hook(x, labels, y)
        x = self.pool(x)
        return x   

class FC_block_VGG(nn.Module): # First Linear+Act, then Dropout
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init, train_mode):
        super(FC_block_VGG, self).__init__()
        
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.fc(x)
        x = self.act(x)
        if self.dropout != 0:
            x = self.drop(x)
        x = self.hook(x, labels, y)
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