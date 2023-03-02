#!/bin/bash
for dropout in 0 0.1 0.25
do
    for topology in 'FC_500_FC_100' 'FC_1000_FC_100' 'FC_500_FC_500_FC_100' 'FC_1000_FC_1000_FC_100' 'CONV_64_3_1_1_CONV_256_3_1_1_FC_1000_FC_1000_FC_100'
    do
      for train_mode in 'BP' 'FA' 'DFA' 'DRTP'  'sDFA'  'shallow'
      do
        echo $dropout $topology $train_mode 
        python optimal_learning_rate.py --dataset 'CIFAR100' --dropout $dropout --topology $topology --train-mode $train_mode
      done
    done
    for train_mode in 'BP' 'FA' 'DFA' 'DRTP'  'sDFA'  'shallow'
    do
      python optimal_learning_rate.py --dataset 'CIFAR100' --dropout $dropout --topology 'CONV_64_3_1_1_CONV_256_3_1_1_FC_1000_FC_1000_FC_100' --train-mode $train_mode --freeze-conv-layers
    done
done
