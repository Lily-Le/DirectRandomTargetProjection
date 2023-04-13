import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter

import json
import argparse
import train_SNN as train
import setup
import os
import models_SNN as models
import sys
# import torchsummary
import torchinfo
import time
import pdb
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate, layer

VGG16S_topo='CONV2S_64_3_1_1_CONV2S_128_3_1_1_CONV3S_256_3_1_1_CONV3S_512_3_1_1_CONV3S_512_3_1_1_FCV_4096_FCVS_4096_FCVS_10'
def GridSearch(args, device, train_loader, traintest_loader,  test_loader, paramGrid):
    best_score = 0
    max_score = 0
    if args.topology == VGG16S_topo:
        topology_name='VGG16S'
    else:
        topology_name=args.topology
    for lr in paramGrid:
        if args.freeze_conv_layers:
            filepath=args.dataset+'/'+topology_name+'_random'+f'/{args.loss}'+f'/{args.surrogate}/'+args.train_mode+'/'+str(args.dropout)+'/'+str(args.batch_size)+'/'+str(args.optimizer)+f'/T{args.spike_window}'+f'/tau{args.tau}'+'/'+str(lr)
            writer = SummaryWriter('logs_lr_all/'+filepath)
        else:
            filepath=args.dataset+'/'+topology_name+f'/{args.loss}'+f'/{args.surrogate}'+'/'+args.train_mode+'/'+str(args.dropout)+'/'+str(args.batch_size)+'/'+str(args.optimizer)+f'/T{args.spike_window}'+f'/tau{args.tau}'+'/'+str(lr)
            writer = SummaryWriter('logs_lr_all/'+filepath)         
        
        save_path='output_lr_all/'+filepath
        checkpoint_path=save_path+'/checkpoints'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.manual_seed(42)

        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, loss=args.loss, device=device,tau=args.tau,spike_window=args.spike_window,surrogate_=args.surrogate)
        tmp_=sys.stdout
        # filepath = 'logs_lr_all/'+args.dataset+'/'+args.topology+'/'+args.train_mode+'/'+str(args.dropout)+'/'+str(args.batch_size)
        ff = open('logs_lr_all/'+filepath+f'/model_summary_{args.batch_size}.log','w')
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

        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
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
   
        train_scores = []
        train_losses = []
        test_scores = []
        test_losses = []
        patience = args.patience
      
        # graph_inputs=torch.rand(args.batch_size,args.input_channels,args.input_size,args.input_size).type(torch.FloatTensor).cuda()
        # graph_labels=torch.rand(args.batch_size,args.label_features).type(torch.FloatTensor).cuda()
        # writer.add_graph(model,(graph_inputs,graph_labels))
        # pdb.set_trace()
        if (args.cont!=0) and os.path.exists(checkpoint_path+f'/{args.cont}.pth') :

            checkpoint = torch.load(checkpoint_path+f'/{args.cont}.pth')
            model.load_state_dict(checkpoint['model'])
            try:
                # pdb.set_trace()
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('load optimizer state dict')
            except:
                # pdb.set_trace()
                keys_=[*checkpoint['optimizer']['state']]
                for i in range(int(len(keys_)/2)):
                    n1=keys_[i]
                    n2=keys_[i+32]
                    # pdb.set_trace()
                    optimizer.state_dict()['state'][n1]=checkpoint['optimizer']['state'][n2]
                # for n in optimizer.state_dict()['param_groups'][0]['params']:
                #     optimizer.state_dict()['state'][n]=checkpoint['optimizer']['state'][n+32]
                # optimizer.add_param_group({'params':model.parameters()})
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # print('load optimizer')
            # optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
            # del optimizer_tmp
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
            args.cont=0
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')
            
        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,)) 

        # writer.add_graph(model,input_to_model = torch.rand(args.batch_size,args.input_channels,args.input_size,args.input_size))
        train_time_list=[]
        test_time_list=[]
        for epoch in range(start_epoch, args.epochs + 1):
            # Training
            since=time.time()
            train.train_epoch(args, model, device, train_loader, optimizer, loss)
            train_time_list.append(time.time()-since)
            # epochloss = train.eval_epoch(args, model, device,  traintest_loader, loss, 'Train', epoch, writer)
            # Compute accuracy on testing set
            since=time.time()
            trainepochloss, trainscore = train.eval_epoch(args, model, device, traintest_loader, loss, 'Train', epoch, writer)
            testepochloss, testscore = train.eval_epoch(args, model, device, test_loader, loss, 'Test', epoch, writer)
            test_time_list.append(time.time()-since)
            test_losses.append(testepochloss)
            test_scores.append(testscore)
            train_losses.append(trainepochloss)
            train_scores.append(trainscore)
            # early stopping
            if (epoch-start_epoch > 1):
                lossDecrement = (test_losses[epoch-start_epoch-2]-testepochloss)/test_losses[epoch-start_epoch-2]
                print(test_losses[epoch-start_epoch-2])
                print(testepochloss)
                print(lossDecrement)
                if lossDecrement < args.tolerance:
                    patience -= 1
                    if patience == 0:
                        break
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},os.path.join(checkpoint_path,f'latest.pth'))
            if (epoch % args.ckpt_interval)==0 and (epoch!=0):
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},os.path.join(checkpoint_path,f'{epoch}.pth'))
                # torch.save(model.state_dict(), os.path.join(save_path,f'latest.pth'))
                print(f'Model saved! Epoch= {epoch}')
        
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}, os.path.join(checkpoint_path,f'{epoch}.pth'))
        print(f'Model saved! Epoch= {epoch}')
        final_score = np.mean(test_scores[-10:])
        
        # Save training and testing time
        write_json(train_time_list,save_path+'/train_time.json')
        write_json(test_time_list,save_path+'/test_time.json')
        write_json(test_losses,save_path+'/test_loss.json')
        write_json(test_scores,save_path+'/test_acc.json')
        write_json(train_losses,save_path+'/train_loss.json')
        write_json(train_scores,save_path+'/train_acc.json')        
        if final_score > best_score:
            optimal_learning_rate = lr 
            best_score = final_score
        
        final_score2 = np.max(test_scores)
        if final_score2 > max_score:
            optimal_learning_rate2 = lr 
            max_score = final_score2

    return optimal_learning_rate, optimal_learning_rate2


def write_json(var_list,file_name):
    with open(file_name, 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(var_list, indent=2, ensure_ascii=False))
    



def main():
    parser = argparse.ArgumentParser(description='Training fully-connected and convolutional networks using backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), and direct random target projection (DRTP)')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug', 'CIFAR100','IMAGENET','IMAGENETTE'], default='CIFAR10', help='Choice of the dataset: synthetic regression (regression_synth), synthetic classification (classification_synth), MNIST (MNIST), CIFAR-10 (CIFAR10), CIFAR-10 with data augmentation (CIFAR10aug). Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: MNIST.')
    parser.add_argument('--data-path',type=str,default='/home/cll/data/imagenette/',help='ImageNet Data Root Path')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='FA', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='Adam', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='MSE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--trials', type=int, default=1, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=64, help='Input batch size for testing Default: 1000.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4.')
    # Network  #CONV_32_1_2_FC_1000_FC_100
    parser.add_argument('--topology', type=str, default='CONVS_32_5_1_2_FCS_1000_FCS_10', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')
    # parser.add_argument('--codename', type=str, default='test')
    
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Early stopping. Default: 1e-3.')
    parser.add_argument('--patience', type=float, default=50, help='Early stopping. Default:10.')
    parser.add_argument('--param-grid', type=float,  nargs='+', help='grid search lr')
    parser.add_argument('--device', type=int,   help='device')
    parser.add_argument('--spike-window', type=int, default=20,   help='window T')
    parser.add_argument('--surrogate', type=str, default='atan',   help='window T')
    parser.add_argument('--tau', type=float, default=2.0,   help='window T')

    
    parser.add_argument('--cont', type=int,default=0,help='Epoch to continue trraining. Default:0, start from the beginning.')
    parser.add_argument('--start-trial', type=int,default=1,help='Starting trial')
    parser.add_argument('--ckpt-interval', type=int,default=50)
    args = parser.parse_args()
    if args.freeze_conv_layers:
        args.codename = args.dataset+'-'+args.topology+'-'+args.train_mode+'-'+str(args.dropout)+'-random'
    else:
        args.codename = args.dataset+'-'+args.topology+'-'+args.train_mode+'-'+str(args.dropout)

    # Generate dataset for classification
    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)
    
    # if args.device is not None:
    #     device=args.device
    # param_grid = [5e-5, 1.5e-5,5e-6]    
    param_grid = [1.5e-3, 5e-4, 1.5e-4, 5e-5, 1.5e-5,5e-6]
    param_grid = [1.5e-3, 5e-4]
    #param_grid = [5e-4,1.5e-4, 5e-5, 1.5e-5,5e-6]

    # param_grid=[1.5e-5,5e-6]
    # param_grid = [1.5e-6, 3e-7, 3e-8, 5e-7]# 
    # param_grid=args.param_grid
    # Create a GridSearchCV object to find the optimal learning rate
    best_params, best_params2 = GridSearch(args, device, train_loader, traintest_loader,  test_loader, param_grid)

    # Print the optimal learning rate
    print("Optimal learning rate:", best_params)

    filepath = 'output_lr_all'
    file = open(filepath+f'/learning_rate_{args.batch_size}.txt','a+')
    if args.freeze_conv_layers:
        file.write(args.dataset+' '+args.topology+'_random '+args.train_mode+' '+str(args.dropout)+' '+str(best_params)+' '+str(best_params2)+'\n')
    else:
        file.write(args.dataset+' '+args.topology+' '+args.train_mode+' '+str(args.dropout)+' '+str(best_params)+' '+str(best_params2)+'\n')
    file.close()



if __name__ == '__main__':
    main()

