#!/usr/bin/env python3
# %%
import glob
import sys
import psutil
import torch
import torch.utils.data as data
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from datetime import datetime
import time
from dataset import dataset
from prefetcher import data_prefetcher
import argparse
import numpy as np

# from radam import RAdam
# from lookahead import Lookahead

# from model_bn import UNet_BN
# from model_in import UNet_IN
# from model_in_ws import UNet_INws
# from model_gn import UNet_GN
# from model_gn_ws import UNet_GNws
# from model_sn import UNet_SN
from model_sn_ws import UNet_SNws
# from model_sn_ws_swish import UNet_SNwsSwish
from model_sn_ws_mish import UNet_SNwsMish

from gc_sgd import SGDgc
from gc_adamw import AdamWgc
from gc_radam import RAdamgc

from lr_scheduler import CosineAnnealingLR


# %%
def get_args():
    # Get the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/fcn/test-set/')
    parser.add_argument('--model_path', default='model/')
    parser.add_argument('--train_f', default='train.lst')
    parser.add_argument('--val_f', default='val.lst')
    parser.add_argument('--tbs', type=int, default=8)
    parser.add_argument('--ag_step', type=int, default=1)
    parser.add_argument('--tbs_target', type=int, default=8)

    parser.add_argument('--vbs', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--anneal_start', type=int, default=30)
    parser.add_argument('--save_epoch', type=int, default=0)
    parser.add_argument('--log_batch', type=int, default=0)

    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--dtoSGD', type=int, default=False)
    parser.add_argument('--lookahead', type=int, default=False)
    parser.add_argument('--la_steps', type=int, default=5)
    parser.add_argument('--la_alpha', type=float, default=0.8)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--amsgrad', type=int, default=False)

    parser.add_argument('--lr0', type=float, default=1e-2)
    parser.add_argument('--warmup')  # constant, gradual
    parser.add_argument('--warmup_step', type=int)
    parser.add_argument('--lr1', type=float)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--nag', type=int, default=1)

    parser.add_argument('--n_channels', type=int, default=14)
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=6)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--norm_layer', default='none')
    parser.add_argument('--num_groups', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=0)
    parser.add_argument('--in_with_mom', type=int, default=0)
    parser.add_argument('--in_mom', type=float, default=0.1)
    parser.add_argument('--affine', type=int, default=1)
    parser.add_argument('--using_movavg', type=int, default=1)
    parser.add_argument('--using_bn', type=int, default=1)
    parser.add_argument('--leps', type=int, default=1)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--wstd', type=int, default=0)
    parser.add_argument('--act_layer', default='relu')
    parser.add_argument('--gc', type=int, default=0)
    parser.add_argument('--gcc', type=int, default=1)
    parser.add_argument('--gcloc', type=int, default=0)

    parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))
    parser.add_argument('--gpu', type=int, default=torch.cuda.is_available())
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    return args


# %%
def main():
    args = get_args()

    if args.resume is not None:
        resume = args.resume.split('.')[0]
        epoch0 = int(resume.split('-')[-1])
        print('Train the model from checkpoint: %s, start epoch: %d' % (resume.split('/')[-1], epoch0))
    else:
        epoch0 = 0
        print('Train the model from scratch.')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    print('Data path: %s' % args.data_path)
    print('Model path: %s' % args.model_path)
    print('Train files: %s' % args.train_f)
    print('Validation files: %s' % args.val_f)

    print('Batch size (train): %d' % args.tbs)
    print('Gradient accumulation step: %d' % args.ag_step)
    print('Batch size (validation): %d' % args.vbs)

    print('Max epochs: %d' % args.max_epochs)
    print('Anneal start: %d' % args.anneal_start)
    print('Checkpoint epoch: %d' % args.save_epoch)
    print('Log batch: %d' % args.log_batch)

    print('Optimizer: %s' % args.optimizer)
    if args.optimizer == 'radam' or args.optimizer == 'ralamb':
        print('Degenerate to SGD: %d' % args.dtoSGD)
    print('Lookahead: %d' % args.lookahead)
    if args.lookahead:
        print('Lookahead steps: %d' % args.la_steps)
        print('Lookahead alpha: %f' % args.la_alpha)
    print('Weight decay: %f' % args.wd)
    print('AMSgrad: %d' % args.amsgrad)

    print('Initial learning rate: %f' % args.lr0)
    if args.warmup is not None:
        print('Warm up mode: %s' % args.warmup)
        print('Warm up step: %d' % args.warmup_step)
        print('Target learning rate: %f' % args.lr1)
    print('Momentum: %f' % args.mom)
    print('NAG: %d' % args.nag)

    print('Input channels: %d' % args.n_channels)
    print('Input filters: %d' % args.n_filters)
    print('Output classes: %d' % args.n_class)
    print('Image size HxW: %dx%d' % (args.H, args.W))
    print('Normalization layer: %s' % args.norm_layer)
    if args.norm_layer == 'gn' and args.num_groups != 0:
        print('Num groups: %d ' % args.num_groups)
    elif args.norm_layer == 'gn' and args.group_size != 0:
        print('Group size: %d ' % args.group_size)
    if args.norm_layer == 'in':
        print('Track running state for inorm: %d ' % args.in_with_mom)
        if args.in_with_mom == 1:
            print('Momentum for inorm: %f ' % args.in_mom)
        print('Affine: %d ' % args.affine)
    if args.norm_layer == 'sn':
        print('Using moving average: %d ' % args.using_movavg)
        print('Using batch normalization: %d ' % args.using_bn)
    if args.norm_layer == 'frn':
        print('Using learnable_eps: %d ' % args.leps)
        print('eps: %f ' % args.eps)
    if args.norm_layer == 'mabn':
        print('Batch size target (train): %d' % args.tbs_target)
    if args.norm_layer != 'none':
        print('Using weight standarization: %d ' % args.wstd)
    print('Activation function: %s ' % args.act_layer)
    print('Use gradient centralization: %d ' % args.gc)
    print('Use gradient centralization in convolution only: %d ' % args.gcc)
    print('Use local gradient centralization: %d ' % args.gcloc)

    print('num_workers = %d' % args.nproc)
    print('use_cuda = %d' % args.gpu)
    if args.seed is not None:
        print('Random seed: %d' % args.seed)

    # %%
    # load the file lists
    train_lst = args.data_path+args.train_f  # 'train.lst'
    val_lst = args.data_path+args.val_f  # 'val.lst'
    with open(train_lst, 'r') as f:
        train_files = f.read().splitlines()
    with open(val_lst, 'r') as f:
        val_files = f.read().splitlines()
    train_size, val_size = len(train_files), len(val_files)
    print('train : validation = %d : %d' % (train_size, val_size))

    # %%
    # define dataset and dataloader
    train_ds = dataset(train_files)
    val_ds = dataset(val_files)

    train_loader = data.DataLoader(
        train_ds, batch_size=args.tbs, shuffle=False,
        sampler=data.RandomSampler(train_ds, replacement=False),
        num_workers=args.nproc, pin_memory=args.gpu, drop_last=False)
    val_loader = data.DataLoader(
        val_ds, batch_size=args.vbs,
        sampler=data.SequentialSampler(val_ds),
        num_workers=args.nproc, pin_memory=args.gpu, drop_last=False)

    # Build the net
    if args.gpu:
        if args.norm_layer == 'bn':
            net = UNet_BN(args.n_channels, args.n_filters, args.n_class).cuda()
        elif args.norm_layer == 'frn':
            if args.act_layer == 'tlu':
                net = UNet_FRNtlu(args.n_channels, args.n_filters, args.n_class, args.eps, args.leps).cuda()
            elif args.act_layer == 'relu':
                net = UNet_FRNrelu(args.n_channels, args.n_filters, args.n_class, args.eps, args.leps).cuda()
            elif args.act_layer == 'swish':
                net = UNet_FRNswish(args.n_channels, args.n_filters, args.n_class, args.eps, args.leps).cuda()
            elif args.act_layer == 'mish':
                net = UNet_FRNmish(args.n_channels, args.n_filters, args.n_class, args.eps, args.leps).cuda()
        elif args.norm_layer == 'in':
            if args.wstd == 0:
                net = UNet_IN(args.n_channels, args.n_filters, args.n_class, args.in_with_mom, args.in_mom, args.affine).cuda()
            elif args.wstd == 1:
                net = UNet_INws(args.n_channels, args.n_filters, args.n_class, args.in_with_mom, args.in_mom, args.affine).cuda()
        elif args.norm_layer == 'gn':
            if args.wstd == 0:
                net = UNet_GN(args.n_channels, args.n_filters, args.n_class, args.num_groups, args.group_size, args.affine).cuda()
            elif args.wstd == 1:
                net = UNet_GNws(args.n_channels, args.n_filters, args.n_class, args.num_groups, args.group_size, args.affine).cuda()
        elif args.norm_layer == 'sn':
            if args.wstd == 0:
                net = UNet_SN(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            elif args.wstd == 1:
                if args.act_layer == 'swish':
                    net = UNet_SNwsSwish(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
                elif args.act_layer == 'mish':
                    net = UNet_SNwsMish(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
                else:
                    net = UNet_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
        elif args.norm_layer == 'mabn':
            net = UNet_MABN(args.n_channels, args.n_filters, args.n_class, args.tbs, args.tbs_target).cuda()

    # else:
    #     net = UNet(args.n_channels, args.n_filters, args.n_class, args.norm_layer, args.num_groups, args.group_size, args.in_with_mom, args.in_mom, args.affine, args.using_movavg, args.using_bn, args.wstd, args.leps)
    summary(net, input_size=(args.n_channels, args.H, args.W))

    # %%
    # Train
    loss_fun = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = SGDgc(net.parameters(), lr=args.lr0, momentum=args.mom, dampening=0,
                          weight_decay=args.wd, nesterov=args.nag,
                          use_gc=args.gc, gc_conv_only=args.gcc)
    elif args.optimizer == 'adamw':
        optimizer = AdamWgc(net.parameters(), lr=args.lr0, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=args.wd, amsgrad=args.amsgrad,
                            use_gc=args.gc, gc_conv_only=args.gcc, gc_loc=args.gcloc)
    elif args.optimizer == 'radam':
        optimizer = RAdamgc(net.parameters(), lr=args.lr0, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=args.wd, amsgrad=args.amsgrad, degenerated_to_sgd=args.dtoSGD,
                            use_gc=args.gc, gc_conv_only=args.gcc, gc_loc=args.gcloc)
    # elif args.optimizer == 'ralamb':
    #     optimizer = RALamb(net.parameters(), lr=args.lr0, betas=(0.9, 0.999), eps=1e-8,
    #                        weight_decay=args.wd, amsgrad=args.amsgrad, degenerated_to_sgd=args.dtoSGD)
    if args.lookahead:
        optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)

    # load model and optimizer to resume training
    if epoch0 != 0:
        print('Loading model checkpoint: %s' % (resume+'.ckp'))
        print('Loading optimizer checkpoint: %s' % (resume+'.opt'))
        optimizer.load_state_dict(torch.load(resume+'.opt'))
        if args.gpu:
            # net.load_state_dict(torch.load(resume+'.ckp'))
            net = torch.load(resume+'.ckp')
        else:
            # net.load_state_dict(torch.load(resume+'.ckp', map_location=torch.device('cpu')))
            net = torch.load(resume+'.ckp', map_location=torch.device('cpu'))

    # start training
    timestamp = datetime.now()
    print('[%s] Start training/validation.' % timestamp)
    model_name = args.model_path+'model-'+'{0:%Y-%m-%d-%H-%M-%S-%f}'.format(timestamp)
    best = {'accu': 0.0, 'epoch': epoch0}

    scheduler = CosineAnnealingLR(optimizer, args.anneal_start, args.max_epochs-args.anneal_start, eta_min=1e-8, last_epoch=-1, verbose=True)
    for epoch in range(epoch0, epoch0+args.max_epochs):
        t0 = time.time()

        # set learning rate warm up
        if (epoch0 == 0) and (args.warmup is not None):
            lr = adjust_warmup_lr(epoch, args.lr0, args.lr1, args.warmup_step, args.warmup)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # switch to training mode
        net.train()
        prefetcher = data_prefetcher(train_loader, args.gpu)
        inputs, targets, index = prefetcher.next()
        i = 0
        optimizer.zero_grad()
        while (inputs is not None) and (targets is not None):
            outputs = net(inputs)
            loss = loss_fun(outputs, targets)  # averaged across observations for each minibatch
            # print(torch.sum(inputs != inputs))
            # print(torch.sum(outputs != outputs))
            # print(torch.sum(targets != targets))
            # print(loss, outputs.shape, targets.shape)

            n_samples = targets.shape[0]  # current batch size

            # accumulate gradient and update net
            loss = loss/args.ag_step
            loss.backward()
            if ((i+1) % args.ag_step == 0) or (n_samples < args.tbs):
                optimizer.step()
                optimizer.zero_grad()  # zero the gradient buffers

            # prefetch train data
            inputs, targets, index = prefetcher.next()
            i += 1

        # switch to evaluate mode
        if args.lookahead:
            optimizer._backup_and_load_cache()

        if (args.norm_layer == 'sn') and (args.using_movavg == False):
            sn_helper(train_loader, net)

        net.eval()
        # evaluate training set
        train_loss, train_correct, train_count = 0.0, 0.0, 0
        prefetcher = data_prefetcher(train_loader, args.gpu)
        with torch.no_grad():
            inputs, targets, _ = prefetcher.next()
            j = 0
            while (inputs is not None) and (targets is not None):
                outputs = net(inputs)  # with shape NCHW
                loss = loss_fun(outputs, targets)  # averaged across observations for each minibatch

                n_samples = targets.shape[0]  # current batch size
                train_loss += loss.item() * n_samples#!/usr/bin/env python3
                train_count += n_samples
                # print(torch.sum(inputs != inputs))
                # print(torch.sum(outputs != outputs))
                # print(torch.sum(targets != targets))
                # print(train_loss, train_count, n_samples, loss, outputs.shape, targets.shape)

                _, predict = torch.max(outputs.data, 1)  # with shape NHW
                train_correct += (predict == targets).sum().item()/args.H/args.W  # total for each minibatch

                # prefetch train data
                inputs, targets, _ = prefetcher.next()
                j += 1

        # evaluate validation set
        val_correct, val_count = 0.0, 0
        prefetcher = data_prefetcher(val_loader, args.gpu)
        with torch.no_grad():
            inputs, targets, _ = prefetcher.next()
            j = 0
            while (inputs is not None) and (targets is not None):
                outputs = net(inputs)  # with shape NCHW
                _, predict = torch.max(outputs.data, 1)  # with shape NHW
                val_correct += (predict == targets).sum().item()/args.H/args.W
                val_count += targets.shape[0]

                # prefetch train data
                inputs, targets, _ = prefetcher.next()
                j += 1

        # print statistics
        train_accu = train_correct/train_count
        val_accu = val_correct/val_count
        print('[%3d/%3d] lr: %f    train_loss: %f    train_accu: %f    val_accu: %f    running_time: %.3f s' %
              (epoch+1, args.max_epochs,
               optimizer.param_groups[0]['lr'],
               train_loss/train_count,
               train_accu,
               val_accu,
               time.time()-t0))

        # save checkpoint
        if args.save_epoch and (epoch % args.save_epoch == (args.save_epoch-1)):
            torch.save(net, '%s-%d.ckp' % (model_name, epoch+1))
            torch.save(optimizer.state_dict(), '%s-%d.opt' % (model_name, epoch+1))
            print('[%s] Checkpoint saved to %s-%d' % (datetime.now(), model_name, epoch+1))

        # save currently best model
        if best['accu'] < val_accu:
            best['accu'] = val_accu
            best['epoch'] = epoch+1
            torch.save(net, model_name+'-best.pth')

        if args.lookahead:
            optimizer._clear_and_load_backup()

        scheduler.step()

    print('[%s] Finished training/validation. Best accuracy %f @ epoch %d' % (datetime.now(), best['accu'], best['epoch']))

def adjust_warmup_lr(epoch, lr_init, lr_target, warmup_step, warmup_mode):
    if warmup_mode == 'constant':
        if epoch < warmup_step-1:
            lr = lr_init
        else:
            lr = lr_target
    elif warmup_mode == 'gradual':
        if epoch < warmup_step-1:
            k = (lr_target-lr_init)/(warmup_step-1)
            lr = lr_init+k*epoch
        else:
            lr = lr_target
    return lr


def sn_helper(train_loader, model):
    ITER_COMPUTE_BATCH_AVEARGE = 200

    model.train()

    for name, param in model.state_dict().items():
        if 'running_mean' in name:
            param.fill_(0)
        elif 'running_var' in name:
            param.fill_(0)

    prefetcher = data_prefetcher(train_loader, use_cuda=True)
    with torch.no_grad():
        inputs, _, _ = prefetcher.next()
        i = 0
        # for i, (input, target) in enumerate(train_loader):
        while (inputs is not None):  # and (targets is not None):
            if i == ITER_COMPUTE_BATCH_AVEARGE:
                break
            # target = target.cuda(non_blocking=True)
            model(inputs)
            # prefetch train data
            inputs, _, _ = prefetcher.next()
            i += 1

    for name, param in model.state_dict().items():
        if 'running_mean' in name:
            param /= ITER_COMPUTE_BATCH_AVEARGE
            model.state_dict()[name.replace('running_mean', 'running_var')] /= ITER_COMPUTE_BATCH_AVEARGE
            model.state_dict()[name.replace('running_mean', 'running_var')] -= param ** 2


if __name__ == '__main__':
    main()
