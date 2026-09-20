#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/9/26 18:04
# @Author: hb925
# @File: run.py
import argparse
import random

import numpy as np
import torch
from torch_kt.runner import main
from torch_kt.runner_wandb import wandb_main
from torch_kt.wandb_utils import get_wandb_config


parser = argparse.ArgumentParser(description='train model')
parser.add_argument('model_name', nargs='?', default='DKT', help='Model name: assist0910,assist2012')
parser.add_argument('data_name', nargs='?', default='assist0910', help='Dataset name: assist0910,assist2012')
parser.add_argument('--logs_base', default='./logs', help='logs dir path')
parser.add_argument('--data_base', default='./data', help='logs dir path')
# parser.add_argument('--emb_type',type=str, default='', choices=['sep', 'opp','cross','pre'])#要结合对应模型修改
parser.add_argument('--folds', type=int, default=-1, help='cross evaluate folds')
parser.add_argument('--max_len', type=int, default=100, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda','npu'])
parser.add_argument('--device_id', type=int, default='0')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs for training')  ## 500
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--weight_decay', type=float, default=0, help='l2')
parser.add_argument('--valid_interval', type=int, default=1, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--remote', action='store_true', default=False, help='use wandb')
parser.add_argument('--clean_weights', action='store_true', default=False, help='sweep config file')

parser.add_argument('--emb_size', type=int, default=64, help='embedding size of input')
parser.add_argument('--hidden_units', type=int, default=128, help='size of kt hidden state')

parser.add_argument('--attention_blocks', type=int, default=2, help='self-attention block nums:SAKT AKT SAINT')
parser.add_argument('--att_heads', type=int, default=4, help='self-attention head nums:SAKT AKT SAINT')

parser.add_argument('--dff', type=int, default=256, help='dff for AKT ')
parser.add_argument('--final_fc_dim', type=int, default=256, help='final_fc_dim for AKT')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 loss weight for AKT')

parser.add_argument('--mlp_hidden', type=int, default=128, help='mlp_hidden for KQN')
parser.add_argument('--rnn_hidden', type=int, default=128, help='rnn_hidden for KQN')

parser.add_argument('--memory_dim', type=int, default=128, help='memory_dim for DKVM Deep-IRT')
parser.add_argument('--memory_size', type=int, default=64, help='memory_size for DKVM Deep-IRT')

parser.add_argument('--answer_dim', type=int, default=8, help='embedding size of response')
parser.add_argument('--attention_dim', type=int, default=64, help='attention_dim for ATKT')

parser.add_argument('--proj', action='store_true', default=False, help='proj for DTransformer')
parser.add_argument('--hard_neg', action='store_true', default=True, help='hard_neg for DTransformer')
parser.add_argument('--shortcut', action='store_true', default=False, help='shortcut for DTransformer')
parser.add_argument('--use_cl', action='store_true', default=False, help='use_cl for DTransformer')
parser.add_argument('--n_layers', type=int, default=1,choices=[1,2,3], help='n_layers for DTransformer')
parser.add_argument('--lambda_cl', type=float, default=1.0, help='lambda_cl for DTransformer')
parser.add_argument('--n_know', type=int, default=16, help='n_layers for DTransformer')

parser.add_argument('--separate_qa', action='store_true', default=False, help='separate_qa for SparseKT')
parser.add_argument('--sparse_ratio', type=float, default=0.8, help='sparse_ratio for SparseKT')
parser.add_argument('--k_index', type=int, default=16, help='k_index for SparseKT')

parser.add_argument('--penumbra_r', type=float, default=0.1, help='penumbra_r for StableKT')
parser.add_argument('--penumbra_g', type=float, default=0.1, help='penumbra_g for StableKT')
parser.add_argument('--max_distance', type=int, default=50, help='max_distance for StableKT')
parser.add_argument('--num_buckets', type=int, default=32, help='num_buckets for StableKT')
parser.add_argument('--seed', type=int, default=100, help='the number of random seed')
parser.add_argument('--fast_run', action='store_true', default=False,
                    help='fast run in sample data')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("warning:cuda is not enable!!!!!!!!!!!!!!!!!!")
            args.device = "cpu"
        else:
            args.device = f'{args.device}:{args.device_id}'
    elif args.device == "npu":
        import torch_npu
        if not torch_npu.npu.is_available():
            print("warning:npu is not enable!!!!!!!!!!!!!!!!!!")
            args.device = "cpu"
        else:
            args.device = f'{args.device}:{args.device_id}'
            torch_npu.npu.set_device(args.device_id)
    run_params = vars(args)
    remote = run_params.pop("remote")
    if remote:
        get_wandb_config("configs/config.json")
        wandb_main(**vars(args))
    else:
        main(**vars(args))
