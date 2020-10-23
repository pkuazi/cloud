#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20/04/2018 4:45 PM
# @Author  : Kris
# @Site    :
# @File    : modules.py
# @Software: PyCharm
# @describe:

import numpy as np
import pandas as pd
import time, os, sys, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
from config import config
import gdal
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as functional
import cv2
import tifffile as tiff

random.seed(20180122)
np.random.seed(20180122)


class ImgDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.argmax(y, axis=1)  # 将label改装成(-1,512,512)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_x = torch.FloatTensor(np.array(self.x[idx]))
        idx_y = torch.LongTensor(np.array(self.y[idx]))
        return idx_x, idx_y

class TileImgdataset(Dataset):
    def __init__(self, config, files_names, shuffle=False):
        random.seed(20201023)
        self.conf = config
        self.img_size_x = config.img_rows
        self.img_size_y = config.img_cols
        self.shuffle = shuffle
        self.file_names = files_names
        self.data_set = []
    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.sample(range(len(self.file_names)), 1)[0]
        filename= self.file_names[idx]
        imgfile = os.path.join(config.tile_img_path, filename)
        gtfile = os.path.join(config.tile_gt_path, filename)
        imgds = gdal.Open(imgfile)
        gtds = gdal.Open(gtfile)
        data_x = imgds.ReadAsArray()
        data_y = gtds.ReadAsArray()
        data_x = torch.torch.FloatTensor(data_x)
        data_y = torch.LongTensor(data_y)
        return data_x, data_y 
    def __len__(self):
        return len(self.file_names)
        
class OrigImgdataset(Dataset):
    def __init__(self, config, files_names, shuffle=False):
        logging.info("ImgloaderPostdam->__init__->begin:")
        random.seed(20201023)
        self.conf = config
        self.img_size_x = config.img_rows
        self.img_size_y = config.img_cols
        self.shuffle = shuffle
        self.file_names = files_names
        self.data_set = []
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.sample(range(len(self.file_names)), 1)[0]
        
        filename, xoff, yoff = self.file_names[idx]
        imgfile = os.path.join(config.img_path, filename)
        gtfile = os.path.join(config.gt_path, filename)
        imgds = gdal.Open(imgfile)
        gtds = gdal.Open(gtfile)
        data_x = imgds.ReadAsArray(xoff, yoff, config.BLOCK_SIZE, config.BLOCK_SIZE)
        data_y = gtds.ReadAsArray(xoff, yoff, config.BLOCK_SIZE, config.BLOCK_SIZE)
        data_x = torch.torch.FloatTensor(data_x)
        data_y = torch.LongTensor(data_y)
        return data_x, data_y 
    
class preddataset(Dataset):
    def __init__(self, file_path=None):
        super(dataset, self).__init__()
        self.img = file_path
#         self.img = [x.replace('/gt/', '/imgs/') for x in self.gt]

    def __getitem__(self, index):
        tiffile, xoff, yoff = self.img[index]
        ds = gdal.Open(tiffile)
        fy4a_tile_data = ds.ReadAsArray(xoff, yoff, config.BLOCK_SIZE, config.BLOCK_SIZE)
        fy4a_tile_data[np.isnan(fy4a_tile_data)] = 0
        input = torch.from_numpy(fy4a_tile_data)

        return input, index

    def __len__(self):
        return len(self.img)

    def filelist(self):
        return self.img         

class predprefetcher():
    def __init__(self, loader, use_cuda=True):
        self.loader = iter(loader)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
#             self.next_inputs, self.next_targets, self.next_index = next(self.loader)
            self.next_inputs, self.next_index = next(self.loader)
        except StopIteration:
            self.next_inputs = None
            self.next_targets = None
            self.next_index = None
            return
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.next_inputs = self.next_inputs.cuda(non_blocking=True)
                self.next_targets = self.next_targets.cuda(non_blocking=True)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
#         targets = self.next_targets
        index = self.next_index
        self.preload()
#         return inputs, targets, index
        return inputs, index

class My_Model():
    def __init__(self, config, base_model):
        self.conf = config
        self.img_size_x = config.img_rows
        self.img_size_y = config.img_cols
#         self.model = base_model.cuda()
        self.model = base_model
        self.criterion = nn.CrossEntropyLoss().cuda()
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.9, 0.999))

    def get_iou(self, output, target):
        # return 0
        pred_y = output.data.cpu().numpy()
        pred_y = np.argmax(pred_y, axis=1)
        real_y = target.data.cpu().numpy()
        assert pred_y.shape == real_y.shape, "MY_MODEL get_iou error"
        # print("get_iou pridict label value discript",set(list(np.argmax(output, axis=1).reshape(-1))))
        sample_num = len(target)
        # print(pred_y.shape,real_y.shape)
        # print("sum pred and real",np.sum(pred_y),np.sum(real_y))
        bing = sample_num * self.img_size_x * self.img_size_x
        # bing =  bing - len(np.where((pred_y == 0) & (real_y == 0))[0])
        jiao = 0.0
        for class_id in range(self.conf.class_num):
            jiao += len(np.where((pred_y == class_id) & (real_y == class_id))[0])
        return jiao, bing

    def evalue(self, val_loader):
        running_acc = 0.0
        running_loss = 0.0
        iou_jiao = 0.0
        iou_bing = 0.0
        sample_num = 0
        for data_x, data_y in val_loader:
            img_count = len(data_y)
            sample_num += img_count
            # print(len*(data_y),np.sum(data_y.numpy()))
            data_x = Variable(data_x.cuda())
            data_y = Variable(data_y.cuda())

            # output is the predicted image ?

            outputs = self.model(data_x)
            loss = self.criterion(outputs, data_y)
            _, preds = torch.max(outputs.data, 1)
            running_loss += (loss.item() * img_count * self.img_size_x * self.img_size_y)
            running_acc += torch.sum(preds == data_y.data).item()
            jiao, bing = self.get_iou(outputs, data_y)
            iou_bing += bing
            iou_jiao += jiao
        val_loss = running_loss / (sample_num * self.img_size_x * self.img_size_y)
        val_acc = running_acc / (sample_num * self.img_size_x * self.img_size_y)
        return val_loss, val_acc, iou_jiao / iou_bing

    def update(self, train_loader):
        running_acc = 0.0
        running_loss = 0.0
        iou_bing = 0.0
        iou_jiao = 0.0
        sample_num = 0
        # 分批训练样本
        for i, (data_x, data_y) in enumerate(train_loader, 1):
            img_count = len(data_y)
            sample_num += img_count
            data_x = Variable(data_x.cuda())
            data_y = Variable(data_y.cuda())
            # data_x = Variable(data_x)
            # data_y = Variable(data_y)
            # forward
            outputs = self.model(data_x)  # 正向传播
            loss = self.criterion(outputs, data_y)  # 使用交叉熵计算损失

            # backward
            self.optimizer_ft.zero_grad()  # 将参数的梯度设置为0
            loss.backward()  # 反向传播
            self.optimizer_ft.step()  # 用优化器更新参数

            # 计算该轮误差
            _, preds = torch.max(outputs.data, 1)  # 计算outputs中每行的最大值
            running_loss += (loss.item() * img_count * self.img_size_x * self.img_size_y)
            running_acc += torch.sum(preds == data_y.data).item()
            jiao, bing = self.get_iou(outputs, data_y)
            iou_bing += bing
            iou_jiao += jiao

        train_loss = running_loss / (sample_num * self.img_size_x * self.img_size_y)
        train_acc = running_acc / (sample_num * self.img_size_x * self.img_size_y)
        return train_loss, train_acc, iou_jiao / iou_bing

    def train(self, train_loader, val_loader, epochsize, early_stop=30):
        log_data = {"time_use": [], "iou": [], "val_iou": [], "acc": [], "loss": [], "val_acc": [], "val_loss": []}
        best_iou = -1e10
        self.best_model = copy.deepcopy(self.model)
        best_distence = 0  # 距离best点的距离
        for epoch in range(epochsize):
            st_time = time.time()
            if best_distence >= early_stop:
                self.model = copy.deepcopy(self.best_model)
                break

            best_distence += 1
            # self.adjustLearningRate(self.optimizer_ft, epoch)  # 调整优化器的学习率
            train_loss, train_acc, train_iou = self.update(train_loader)
            val_loss, val_acc, val_iou = self.evalue(val_loader)
            epoch_log = '[%.2d/%.2d] time:%.3f loss:%.6f, acc:%.6f,iou:%.6f; val_loss:%.6f,val_acc:%.6f,val_iou:%.6f.' % \
                        (epoch, epochsize, time.time() - st_time, train_loss, train_acc, train_iou, val_loss, val_acc,
                         val_iou)

            # 更新维护best值
            if best_iou < val_iou:
                best_iou = val_iou
                best_distence = 0
                self.best_model = copy.deepcopy(self.model)

            print(epoch_log)
            log_data["time_use"].append("%.4f" % (time.time() - st_time))
            log_data["acc"].append("%.4f" % train_acc)
            log_data["loss"].append("%.4f" % train_loss)
            log_data["val_acc"].append("%.4f" % val_acc)
            log_data["val_loss"].append("%.4f" % val_loss)
            log_data["iou"].append("%.4f" % train_iou)
            log_data["val_iou"].append("%.4f" % val_iou)
        self.log2file(log_data)


    def get_f1(self, output, target):
        # return 0
        pred_y = output.data.cpu().numpy()
        pred_y = np.argmax(pred_y, axis=1)
        real_y = target.data.cpu().numpy()
        assert pred_y.shape == real_y.shape, "MY_MODEL get_iou error"
        tp = np.zeros(6)
        C = np.zeros(6)
        P = np.zeros(6)
        for class_id in range(self.conf.class_num):
#            tp[class_id] = np.sum((real_y == class_id) & (pred_y == class_id))
#            C[class_id] = np.sum(real_y == class_id)
#            P[class_id] = np.sum(pred_y == class_id)
            tp[class_id] = len(np.where((real_y == class_id) & (pred_y == class_id))[0])
            C[class_id] = len(np.where(real_y == class_id)[0])
            P[class_id] = len(np.where(pred_y == class_id)[0])

        return tp, C, P

    # 模型预测
    def predict(self, test_loader):
        iou_jiao = 0.0
        iou_bing = 0.0
        test_pred = []

        # test accuracy by zjh
        tp = np.zeros(6)
        C = np.zeros(6)
        P = np.zeros(6)

        for label_x, label_y in test_loader:
            label_x = Variable(label_x.cuda())
            label_y = Variable(label_y.cuda())
            outputs = self.model(label_x)
            test_pred.extend(outputs.data.cpu().numpy())
            jiao, bing = self.get_iou(outputs, label_y)
            iou_bing += bing
            iou_jiao += jiao

        #     test accuracy by zjh
            x,y,z = self.get_f1(outputs, label_y)
            tp += x
            C +=y
            P +=z


        test_pred = np.array(test_pred)
        print("test_iou: %.6f" % (iou_jiao / iou_bing))
        # test_pred = np.transpose(test_pred, (0, 2, 3, 1))

        print('true positives: ', tp)
        print('class pixels: ', C)
        print('predicted pixels: ', P)
        
        recall = np.zeros(6)
        precision = np.zeros(6)
        F1 = np.zeros(6)
        for i in range(6):
            recall[i]=tp[i]*1.0/C[i]
            precision[i] = tp[i]*1.0/P[i]
            F1[i]=2*(recall[i]*precision[i])/(recall[i]+precision[i])
        result = {'Imp.surf.':[recall[0],precision[0],F1[0]], 'Clutter/background':[recall[1],precision[1],F1[1]],'Building':[recall[2],precision[2],F1[2]], 'Low veg.': [recall[3],precision[3],F1[3]], 'Tree':[recall[4],precision[4],F1[4]], 'Car': [recall[5],precision[5],F1[5]]}
        result = pd.DataFrame(result,index=['recall','precision','F1 score'])
#        overall_F1 = 2* tp.sum()/(C.sum()+P.sum())
        # print(F1)
        return test_pred, result

    # torch模型在训练时的学习率调整
    def adjustLearningRate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = 0.05 * (0.85 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_opt(self):
        optimizer = torch.optim.SGD(
            [
                {'params': self.model.conv.parameters()},  # 卷积层参数
                {'params': self.model.fc.parameters(), 'lr': 0.05}  # 全连接层层参数
            ],
            lr=0.0,
            momentum=0.75,
            weight_decay=1e-4,
        )
        return optimizer

    # 将日志输出至文件
    def log2file(self, logs_data):
        log = pd.DataFrame(logs_data);
        if not os.path.exists("./logs"): os.mkdir("./logs")
        log.to_csv("./logs/train_logs.csv", index=False, index_label=False);
