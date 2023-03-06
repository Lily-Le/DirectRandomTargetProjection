#%%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# import torchsummary
import torchinfo
import models
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import sys
from torchviz import make_dot, make_dot_from_trace



torch.manual_seed(42)
topology='FC_500_FC_10'
input_size=32
input_channels=3
batch_size=8
label_features=10
dropout=0
train_mode='DRTP'
output_act='sigmoid'
hidden_act='tanh'
conv_act='tanh'
loss='BCE'
device=0
fc_zero_init=False
# Network topology
model = models.NetworkBuilder(topology, input_size=input_size, input_channels=input_channels, label_features=label_features, train_batch_size=batch_size, train_mode=train_mode, dropout=dropout, conv_act=conv_act, hidden_act=hidden_act, output_act=output_act, fc_zero_init=fc_zero_init, loss=loss, device=device).cuda()
input_shape=(batch_size,input_channels,input_size,input_size)
# rand_input = torch.zeros(input_shape, requires_grad=True)
rand_input=torch.rand(input_shape).to('cuda:0')
rand_labels=torch.rand(batch_size,label_features).to('cuda:0')
#%%
make_dot(model(rand_input,rand_labels), params={**{'x': rand_input, 'labels':rand_labels}, **dict(model.named_parameters())}, show_attrs=True, show_saved=True)

# %%
