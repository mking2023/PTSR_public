import os
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from ptsr import *
from torch.utils.tensorboard import SummaryWriter


torch.set_printoptions(precision=7)
torch.set_printoptions(threshold=10000)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_step', type=int, default=1000)
    parser.add_argument('--lr_decay_rate', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--batchsize_eval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Tools')
    parser.add_argument('--model', type=str, default='PTSR')
    args, _ = parser.parse_known_args()
    
    init_environment(args)
        
    return args, parser
    

@RunTime
def main():
    args, parser = init_args()        
    dataset = data_partition(f'dataset/{args.dataset}/{args.dataset}.csv', 'uid', 'iid', 'time')  # train, valid, test, user_num, item_num
    print(f"user num = {dataset[3]},\nitem num = {dataset[4]}")

    args, model_class = get_model(args, parser)
    args.log_dir = init_log(args) 
    args.user_num = dataset[3] 
    args.item_num = dataset[4]
    print_args(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    model = model_class(args, writer).cuda()
    print_model_parameters(model)
    model.train_model(dataset)
            

if __name__ == '__main__':
    main()
    
