## https://github.com/danielchyeh/ImageNet-100-Pytorch/blob/main/generate_IN100.py
import os
import shutil
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for generating ImageNet-100')

    parser.add_argument('--source_folder', type=str,
     default='/home/wwh/imagenet/val', help='folder of ImageNet-1K dataset')
    parser.add_argument('--target_folder', type=str,
     default='/home/cll/data/imagenet100/val', help='folder of ImageNet-100 dataset')
    parser.add_argument('--target_class', type=str,
     default='IN100.txt', help='class file of ImageNet-100')

    opt = parser.parse_args()

    return opt

f = []
def generate_data(source_folder, target_folder, target_class):

    txt_data = open(target_class, "r") 
    for ids, txt in enumerate(txt_data):
        s = str(txt.split('\n')[0])
        f.append(s)

    for ids, dirs in enumerate(os.listdir(source_folder)):
        for tg_class in f:
            if dirs == tg_class:
                print('{} is transferred'.format(dirs))
                shutil.copytree(os.path.join(source_folder,dirs), os.path.join(target_folder,dirs)) 

def main():
    opt = parse_option()
    generate_data(opt.source_folder, opt.target_folder, opt.target_class)

if __name__ == "__main__":
    main()
