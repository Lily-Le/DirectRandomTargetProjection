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

 "main.py" - Main file for training fully-connected and convolutional networks using backpropagation (BP),
    feedback alignment (FA) [Lillicrap, Nat. Comms, 2016], direct feedback alignment (DFA) [Nokland, NIPS, 2016],
    and the proposed direct random target projection (DRTP).
    Example: use the following command to reach ~70% accuracy on the test set of CIFAR-10 using DRTP:
         python main.py --dataset CIFAR10aug --train-mode DRTP --epochs 200 --freeze-conv-layers
                        --dropout 0.05 --topology CONV_64_3_1_1_CONV_256_3_1_1_FC_2000_FC_10
                        --loss CE --output-act none --lr 5e-4
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""


import argparse
import train_0224 as train
import setup
import os


def filedel(filepath):
    for i in ['/testloss.txt','/testacc.txt','/trainloss.txt','/trainacc.txt','/testtime.txt','traintime.txt']:
        try:
            os.remove(filepath+i)
        except:
            pass
            
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

def main():
    parser = argparse.ArgumentParser(description='Training fully-connected and convolutional networks using backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), and direct random target projection (DRTP)')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug', 'CIFAR100','IMAGENET','IMAGENETTE'], default='IMAGENETTE', help='Choice of the dataset: synthetic regression (regression_synth), synthetic classification (classification_synth), MNIST (MNIST), CIFAR-10 (CIFAR10), CIFAR-10 with data augmentation (CIFAR10aug). Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: MNIST.')
    parser.add_argument('--data-path',type=str,default='/home/cll/Workspace/data/cls/imagenette/',help='ImageNet Data Root Path')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='FA', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='Adam', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--trials', type=int, default=10, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=128, help='Input batch size for testing Default: 1000.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4.')
    # Network  #CONV_32_5_1_2_FC_1000_FC_100
    parser.add_argument('--topology', type=str, default='CONVS_32_5_1_2_FCS_1000_FCS_10', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')

    # parser.add_argument('--topology', type=str, default='CONV2_64_3_1_1_CONV2_128_3_1_1_CONV3_256_3_1_1_CONV3_512_3_1_1_CONV3_512_3_1_1_FCV_4096_FCV_4096_FCV_10', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')
    # parser.add_argument('--codename', type=str, default='test')
    parser.add_argument('--cont', type=bool,default=False,help='"Choice the False if retrain from beginning')

    args = parser.parse_args()
    VGG16_topo='CONV2_64_3_1_1_CONV2_128_3_1_1_CONV3_256_3_1_1_CONV3_512_3_1_1_CONV3_512_3_1_1_FCV_4096_FCV_4096_FCV_10'

    if args.topology == VGG16_topo:
        tpg_name='VGG16'
    else:
        tpg_name=args.topology
        
    if args.freeze_conv_layers:
        args.codename = args.dataset+'/'+tpg_name+'_random/'+args.train_mode+f'/bs{args.batch_size}'+f'/{args.optimizer}'+'/'+str(args.dropout)
    else:
        # args.codename = args.dataset+'-'+args.topology+'-'+args.train_mode+'-'+str(args.dropout)
        args.codename = args.dataset+'/'+tpg_name+'/'+args.train_mode+f'/bs{args.batch_size}'+f'/{args.optimizer}'+f'/{args.dropout}'


    mkd(args)
    filepath = 'output/'+args.codename
    file = open(filepath+'/para.txt','w')
    file.write('pid:'+str(os.getpid())+'\n')
    file.write(str(vars(args)).replace(',','\n'))
    file.close()
    if args.cont ==False:
        filedel(filepath)

    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)
    train.train(args, device, train_loader, traintest_loader, test_loader)

if __name__ == '__main__':
    main()