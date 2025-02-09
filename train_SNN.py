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

 "train.py" - Initializing the network, optimizer and loss for training and testing.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# import torchsummary
import torchinfo
# import models
import models_SNN as models
from tqdm import tqdm
from tensorboardX import SummaryWriter
from spikingjelly.clock_driven import neuron, encoding, functional
import os
import sys
import time
import datetime
VGG16S_topo='CONV2S_64_3_1_1_CONV2S_128_3_1_1_CONV3S_256_3_1_1_CONV3S_512_3_1_1_CONV3S_512_3_1_1_FCVS_4096_FCVS_4096_FCVS_10'
# writer = SummaryWriter('logs')

def filedel(filepath):
    for i in ['/testloss.txt','/testacc.txt','/trainloss.txt','/trainacc.txt','/testtime.txt','traintime.txt']:
        try:
            os.remove(filepath+i)
        except:
            pass


def dir_rename(dir):
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # filename = "_".join([basename, suffix]) # e.g. 'mylogfile_120508_171442'
    dst_dir=f'{dir}_last'
    if os.path.exists(dst_dir):
        os.rename(dst_dir,f'{dst_dir}_{suffix}')
    os.rename(dir,dst_dir)
    print(f'Last training result dir renamed to "{dir}_last"')
    
def mkd(args):
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/' + args.codename.split('/')[0])
    except:
        pass
    try:
        # os.mkdir('output/' + args.codename.split('/')[0]+'/'+args.codename)
        os.makedirs('output/'+args.codename)
    except:
        pass

def train(args, device, train_loader, traintest_loader, test_loader):
    # writer = SummaryWriter('logs/'+args.dataset+'/'+args.train_mode)
    log_path='logs/'+args.codename
        
    # if args.freeze_conv_layers:
    #     log_path='logs/'+args.codename
    #     # writer = SummaryWriter('logs/'+args.dataset+'/'+tpg_name+'_random/'+args.train_mode+'/'+str(args.dropout))
    # else:
    #     log_path='logs/'+args.dataset+'/'+tpg_name+'/'+args.train_mode+f'/bs{args.batch_size}'+'/'+str(args.dropout)
        
    writer=SummaryWriter(log_path)
    torch.manual_seed(42)
    filepath = 'output/'+args.codename
    # save_path=os.path.join(filepath,'checkpoints') #checkpoints
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)   
        
    for trial in range(args.start_trial,args.trials+args.start_trial):
        # Network topology
        model=models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, loss=args.loss, device=device,tau=args.tau,spike_window=args.spike_window,surrogate_=args.surrogate)        # print(list(model.named_parameters()))

        tmp_=sys.stdout
        
        ff = open(filepath+f'/model_summary_{args.batch_size}.log','w')
        sys.stdout = ff
        model_info=torchinfo.summary(model,[(args.batch_size,args.input_channels,args.input_size,args.input_size),[args.label_features]])
        print(args.topology)
        print(args.dataset)
        print(f'batch size {args.batch_size}')
        print(model_info)
        ff.close()
        
        sys.stdout = tmp_
        print(model_info)
        

                 
        if args.cuda:
            model.cuda()
        
        if (args.trials > 1):  #Number of training trials
            print('\nIn trial {} of {}'.format(trial,args.trials))
        if (trial == 1):
            print("=== Model ===" )
            print(model)
        
        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")


        save_path=os.path.join(filepath,f'checkpoints/{trial}') #checkpoints
        save_log_path=os.path.join(filepath,f'logs/{trial}')
        
        if (args.cont == 0) and (os.path.exists(save_log_path)):
            dir_rename(save_log_path)
        if (args.cont == 0) and (os.path.exists(save_path)):
            dir_rename(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        if not os.path.exists(save_log_path):
            os.makedirs(save_log_path)  
            
        file = open(save_log_path+'/para.txt','w')
        file.write('pid:'+str(os.getpid())+'\n')
        file.write(str(vars(args)).replace(',','\n'))
        file.close()
        if (args.cont!=0) and os.path.exists(save_path+f'/{args.cont}.pth') :

            checkpoint = torch.load(save_path+f'/{args.cont}.pth')
            model.load_state_dict(checkpoint['model'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('load optimizer state dict')
            except:
                optimizer=checkpoint['optimizer']
                optimizer.add_param_group({'params':model.parameters()})
                print('load optimizer')
            # optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
            # del optimizer_tmp
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')
            
        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,)) 

        for epoch in range(start_epoch, args.epochs + 1):
            # Training
            since=time.time()
            train_epoch(args, model, device, train_loader, optimizer, loss)
            time_elapsed = time.time() - since
            # Compute accuracy on training and testing set
            print("\nSummary of epoch %d:" % (epoch))
            
            filetraintime=writefile(save_log_path, f'/traintime.txt')
            filetraintime.write(str(epoch) + ' ' + str(time_elapsed) + '\n')
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
         
            test_epoch(args, model, device, traintest_loader, loss, 'Train', epoch, writer, trial)
            since=time.time()
            test_epoch(args, model, device, test_loader, loss, 'Test', epoch, writer, trial)
            time_elapsed = time.time() - since
            
            filetesttime=writefile(save_log_path, f'/testtime.txt')
            filetesttime.write(str(epoch) + ' ' + str(time_elapsed) + '\n')
            print('Testinging complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # Save model
            if (epoch % args.ckpt_interval)==0 and (epoch!=0):
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},os.path.join(save_path,f'{epoch}.pth'))
                # torch.save(model.state_dict(), os.path.join(save_path,f'latest.pth'))
                print(f'Model saved! Epoch= {epoch}')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}, os.path.join(save_path,f'{epoch}.pth'))
        print(f'Model saved! Epoch= {epoch}')
        
def train_epoch(args, model, device, train_loader, optimizer, loss):
    model.train()
    
    if args.freeze_conv_layers: #freeze
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False
    
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        if args.regression:
            targets = label
        else:
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0) #one-hot?
        optimizer.zero_grad()
        output = model(data, targets)
        loss_val = loss[0](output, loss[1](targets)) ###################
        loss_val.backward()
        optimizer.step()
        functional.reset_net(model)

def writefile(filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filetestloss = open(filepath+filename , 'a')
    return filetestloss


def test_epoch(args, model, device, test_loader, loss, phase, epoch, writer, trial):
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)
    
    with torch.no_grad():
        for batch_idx,(data, label) in enumerate(tqdm(test_loader)):
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)
            output = model(data, None)
            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            if not args.regression:
                correct += pred.eq(label.view_as(pred)).sum().item()
    
    functional.reset_net(model)
    loss = test_loss / len_dataset
    save_log_path='output/'+args.codename+f'/logs/{trial}'
    if not args.regression:
        acc = 100. * correct / len_dataset
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))
        
        filetestloss = writefile(save_log_path, f'/testloss.txt')
        filetestacc = writefile(save_log_path, f'/testacc.txt')
        filetrainloss = writefile(save_log_path, f'/trainloss.txt')
        filetrainacc = writefile(save_log_path, f'/trainacc.txt')

        if trial == 1:
            if phase == 'Train':
                writer.add_scalar('train_loss', loss, epoch)
                writer.add_scalar('train_acc', acc, epoch)
            if phase == 'Test':
                writer.add_scalar('test_loss', loss, epoch)
                writer.add_scalar('test_acc', acc, epoch)

        if phase == 'Train':
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetrainacc.write(str(epoch) + ' ' + str(acc) + '\n')
        if phase == 'Test':
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetestacc.write(str(epoch) + ' ' + str(acc) + '\n')

    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))
        filetestloss = writefile(save_log_path, f'/testloss.txt')
        filetrainloss = writefile(save_log_path, f'/trainloss.txt')

        if trial == 1:
            if phase == 'Train':
                writer.add_scalar('train_loss', loss, epoch)
            if phase == 'Test':
                writer.add_scalar('test_loss', loss, epoch)

        if phase == 'Train':
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
        if phase == 'Test':
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')



def eval_epoch(args, model, device, test_loader, loss, phase, epoch, writer):
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)
    
    with torch.no_grad():
        for batch_idx,(data, label) in enumerate(tqdm(test_loader)):
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)
            output = model(data, None)
            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            if not args.regression:
                correct += pred.eq(label.view_as(pred)).sum().item()
                
    functional.reset_net(model)
    loss = test_loss / len_dataset
    if not args.regression:
        acc = 100. * correct / len_dataset
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))

        if phase == 'Train':
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('train_acc', acc, epoch)

        if phase == 'Test':
            writer.add_scalar('test_loss', loss, epoch)
            writer.add_scalar('test_acc', acc, epoch)

        
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))

    return loss, acc