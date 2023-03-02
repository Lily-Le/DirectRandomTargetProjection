from subprocess import Popen
import os

filepath = 'output/learning_rate.txt'
f = open(filepath)
lines = f.readlines()
for line in lines:
    wordlist = line.strip().split(' ')
    dataset = wordlist[0]
    train_mode = wordlist[2]
    dropout = wordlist[3]
    lr = wordlist[5]
    word = wordlist[1].split('_')
    if word[-1]=='random':
        topology = wordlist[1].rsplit('_', 1)[0]
        cmd = "python main.py --dataset %s --train-mode %s --dropout %f --lr %f --topology %s --freeze-conv-layers" % (dataset, train_mode, float(dropout), float(lr), topology)
    else:
        topology = wordlist[1]
        print("python main.py --dataset "+dataset+" --train-mode "+train_mode+" --dropout "+dropout+" --lr "+lr+" --topology "+topology)
        # os.system("python main.py --dataset CIFAR100 --train-mode "+train_mode+" --dropout "+dropout+" --lr "+lr+" --topology "+topology)
        cmd = "python main.py --dataset %s --train-mode %s --dropout %f --lr %f --topology %s" % (dataset, train_mode, float(dropout), float(lr), topology)
    Popen(cmd, shell=True).wait()





