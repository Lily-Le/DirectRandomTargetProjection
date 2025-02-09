import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
import argparse
import train
import setup
import os
import models

def GridSearch(args, device, train_loader, traintest_loader,  test_loader, paramGrid):
    best_score = 0
    max_score = 0
    for lr in paramGrid:
        if args.freeze_conv_layers:
            writer = SummaryWriter('logs_lr_all/'+args.dataset+'/'+args.topology+'_random/'+args.train_mode+'/'+str(args.dropout)+'/'+str(lr))
        else:
            writer = SummaryWriter('logs_lr_all/'+args.dataset+'/'+args.topology+'/'+args.train_mode+'/'+str(args.dropout)+'/'+str(lr))

        torch.manual_seed(42)

        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, loss=args.loss, device=device)
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
   
        scores = []
        losses = []
        patience = args.patience
        for epoch in range(1, args.epochs + 1):
            # Training
            train.train_epoch(args, model, device, train_loader, optimizer, loss)
            # epochloss = train.eval_epoch(args, model, device,  traintest_loader, loss, 'Train', epoch, writer)
            # Compute accuracy on testing set
            epochloss, score = train.eval_epoch(args, model, device, test_loader, loss, 'Test', epoch, writer)
            losses.append(epochloss)
            scores.append(score)
            # early stopping
            if epoch > 1:
                lossDecrement = (losses[epoch-2]-epochloss)/losses[epoch-2]
                print(losses[epoch-2])
                print(epochloss)
                print(lossDecrement)
                if lossDecrement < args.tolerance:
                    patience -= 1
                    if patience == 0:
                        break

        final_score = np.mean(scores[-10:])
        if final_score > best_score:
            optimal_learning_rate = lr 
            best_score = final_score
        
        final_score2 = np.max(scores)
        if final_score2 > max_score:
            optimal_learning_rate2 = lr 
            max_score = final_score2

    return optimal_learning_rate, optimal_learning_rate2



def main():
    parser = argparse.ArgumentParser(description='Training fully-connected and convolutional networks using backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), and direct random target projection (DRTP)')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug', 'CIFAR100'], default='CIFAR100', help='Choice of the dataset: synthetic regression (regression_synth), synthetic classification (classification_synth), MNIST (MNIST), CIFAR-10 (CIFAR10), CIFAR-10 with data augmentation (CIFAR10aug). Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: MNIST.')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='DRTP', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='Adam', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--trials', type=int, default=1, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=100, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Input batch size for testing Default: 1000.')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Early stopping. Default: 1e-3.')
    parser.add_argument('--patience', type=float, default=50, help='Early stopping. Default:10.')
    # Network  #CONV_32_5_1_2_FC_1000_FC_100
    parser.add_argument('--topology', type=str, default='CONV_64_3_1_1_CONV_256_3_1_1_FC_1000_FC_1000_FC_100', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')
    parser.add_argument('--cont', type=bool,default=True,help='"Choice the False if retrain from beginning')

    args = parser.parse_args()

    # Generate dataset for classification
    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)
    
    
    param_grid = [1.5e-3, 5e-4, 1.5e-4, 5e-5, 1.5e-5, 5e-6]# 
    # Create a GridSearchCV object to find the optimal learning rate
    best_params, best_params2 = GridSearch(args, device, train_loader, traintest_loader,  test_loader, param_grid)

    # Print the optimal learning rate
    print("Optimal learning rate:", best_params)

    filepath = 'output'
    file = open(filepath+'/learning_rate.txt','a+')
    if args.freeze_conv_layers:
        file.write(args.dataset+' '+args.topology+'_random '+args.train_mode+' '+str(args.dropout)+' '+str(best_params)+' '+str(best_params2)+'\n')
    else:
        file.write(args.dataset+' '+args.topology+' '+args.train_mode+' '+str(args.dropout)+' '+str(best_params)+' '+str(best_params2)+'\n')
    file.close()



if __name__ == '__main__':
    main()

