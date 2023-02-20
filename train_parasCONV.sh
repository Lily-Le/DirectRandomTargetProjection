#!/bin/bash
for dropout in 0 0.1 0.25
do
    # python src/mnist_mlp_train.py --dataset "mnist" --seed $seed
      for train_mode in 'BP' 'FA' 'DFA' 'DRTP'  'sDFA'  'shallow'
      do
        case $train_mode in
          'BP')
          lr=5e-5;;
          'FA')
          lr=1.5e-4;;
          'DFA')
          lr=5e-5;;
          'DRTP')
          lr=5e-4;;
          'sDFA')
          lr=5e-4;;
          'shallow')
          lr=5e-3;;
        esac
        echo $dropout $topology $train_mode $lr
        python main.py --dataset 'MNIST' --freeze-conv-layers --dropout $dropout --topology 'CONV_32_5_1_2_FC_1000_FC_10' --train-mode $train_mode --lr $lr
      done

      for train_mode in 'BP' 'FA' 'DFA' 'DRTP'  'sDFA'
      do
        case $train_mode in
          'BP')
          lr=5e-4;;
          'FA')
          lr=5e-5;;
          'DFA')
          lr=5e-5;;
          'DRTP')
          lr=1.5e-4;;
          'sDFA')
          lr=1.5e-4;;
        esac
        echo $dropout $topology $train_mode $lr
        python main.py --dataset 'MNIST' --dropout $dropout --topology 'CONV_32_5_1_2_FC_1000_FC_10' --train-mode $train_mode --lr $lr
      done
done