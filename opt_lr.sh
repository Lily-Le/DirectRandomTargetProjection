#!/bin/bash
python optimal_learning_rate.py --train-mode DFA --optimizer SGD --epochs 250
python optimal_learning_rate.py --train-mode DRTP --optimizer SGD --epochs 250
