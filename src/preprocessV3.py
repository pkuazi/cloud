#!/usr/bin/env python3

import glob
import argparse
import psutil
import numpy as np
import torch
import torch.utils.data as data
from dataset import dataset
from prefetcher import data_prefetcher
import time

dirs = ['test-set']

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data/fcn/test-set/')
parser.add_argument('--vp', type=float, default=0.2)
parser.add_argument('--tp', type=float, default=0)
parser.add_argument('--gpu', type=int)
parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))
parser.add_argument('--seed', type=int)
parser.add_argument('--dsp', type=float)
args = parser.parse_args()
data_path = args.data_path
val_percent = args.vp
test_percent = args.tp
use_cuda = args.gpu
num_workers = args.nproc
manual_seed = args.seed
dataset_p = args.dsp

print('data_path: %s' % data_path)
print('val_percent: %.2f' % val_percent)
print('test_percent: %.2f' % test_percent)
print('use_cuda: %d' % use_cuda)
print('num_workers = %d' % num_workers)
if manual_seed is not None:
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('manual_seed: %d' % manual_seed)
print('dataset ratio = %f' % dataset_p)

train, val, blacklist = [], [], []
for dir in dirs:
    t0 = time.time()
    lcs = glob.glob(data_path+'gt/*')
    lcs = np.random.permutation(lcs)
    lcs_size = len(lcs)

    ds = dataset(lcs)
    loader = data.DataLoader(
        ds,
        batch_size=1,
        sampler=data.SequentialSampler(ds),
        num_workers=num_workers,
        pin_memory=use_cuda)

    prefetcher = data_prefetcher(loader, use_cuda)
    inputs, targets, index = prefetcher.next()
    lc_final = []
    i = 0
    while (inputs is not None) and (targets is not None) and i < lcs_size*dataset_p:
        inputs_nan = torch.isnan(inputs).any().item()
        targets_nan = torch.isnan(targets).any().item()
        lc_file = lcs[index]
        if inputs_nan or targets_nan:
            blacklist.extend([lc_file])
        else:
            lc_final.extend([lc_file])
        print('[%5d/%5d] %32s    %s    %s' % (i+1, lcs_size, lc_file.split('/')[-1], inputs_nan, targets_nan))

        # prefetch train data
        inputs, targets, index = prefetcher.next()
        i += 1
    print('et: %.6f s' % (time.time()-t0))

    n_val = int(len(lc_final)*val_percent)
    n_train = len(lc_final) - n_val

    train.extend(lc_final[:n_train])
    val.extend(lc_final[n_train:n_train+n_val])
    print('[%15s] train:validation = %d:%d (total: %d)' % (dir, n_train, n_val, len(lc_final)))

with open(data_path+'train.lst', 'w') as f:
    f.writelines("%s\n" % item for item in train)
with open(data_path+'val.lst', 'w') as f:
    f.writelines("%s\n" % item for item in val)
with open(data_path+'blacklist.lst', 'w') as f:
    f.writelines("%s\n" % item for item in blacklist)

print('train:validation:blacklist = %d:%d:%d' % (len(train), len(val), len(blacklist)))

# # validate the lists
# with open(data_path+'train.lst', 'r') as f:
#     x = f.read().splitlines()
# print(x == train)
# with open(data_path+'val.lst', 'r') as f:
#     x = f.read().splitlines()
# print(x == val)
# with open(data_path+'test.lst', 'r') as f:
#     x = f.read().splitlines()
# print(x == test)
