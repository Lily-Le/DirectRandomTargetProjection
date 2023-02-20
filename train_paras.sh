#!/bin/bash
for dropout in 0 0.1 0.25
do
    # python src/mnist_mlp_train.py --dataset "mnist" --seed $seed
    for topology in 'FC_500_FC_10' 'FC_1000_FC_10'
    do
      for train_mode in 'BP' 'FA' 'DFA' 'DRTP'  'sDFA'  'shallow'
      do
        case $train_mode in
          'BP')
          lr=1.5e-4;;
          'FA')
          lr=5e-4;;
          'DFA')
          lr=1.5e-4;;
          'DRTP')
          lr=5e-4;;
          'sDFA')
          lr=1.5e-4;;
          'shallow')
          lr=1.5e-2;;
        esac
        echo $dropout $topology $train_mode $lr
        python main.py --dataset 'MNIST' --dropout $dropout --topology $topology --train-mode $train_mode
      done
    done
done