import numpy as np
import pandas as pd
import os
import glob

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

dataset = 'MNIST'
topology = 'CONV_32_5_1_2_FC_1000_FC_10'
train_mode = 'DRTP'
dropout= '0.0'
freeze_conv_layers = False
epoch = 100

if freeze_conv_layers:
    codename = dataset+'-'+topology+'-'+train_mode+'-'+str(dropout)+'-random'
else:
    codename = dataset+'-'+topology+'-'+train_mode+'-'+str(dropout)

filepath = 'output/'+codename.split('-')[0]+'/'+codename
file = open(filepath+'/testacc.txt')
lines = file.readlines()
err = []
for line in lines:
    wordlist = line.split(' ')
    if wordlist[0]==str(epoch):
        value = number_in_line(line)
        err.append(100-value)

mean = np.mean(err[-10:])
std = np.std(err[-10:])

print("Mean error:", mean)
print("Standard Deviation of error:", std)
