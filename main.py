# -*- coding:utf-8 -*-
# from __future__ import print_function #把下一个新版本的特性导入到当前版本
import os
import numpy as np
import random, logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(20180122)
np.random.seed(20180122)
from datetime import datetime
import torch
import gdal
from utils.gen_tiles_offs import gen_tiles_offs
from torch.utils.data import DataLoader
# import tifffile as tiff
# import cv2
# from modules.models import My_Model
# from modules.models import ImgloaderPostdam_single_channel, ImgloaderPostdam
from modules.models import OrigImgdataset,TileImgdataset
from dataset import dataset,tiledataset
from prefetcher import tiledata_prefetcher
# from modules.fcn import FCN16, FCN8
# from utils import moveFileto, removeDir
from config import config

log_path = "./logs/"
if not os.path.exists(log_path): os.mkdir(log_path)
logging.basicConfig(
    filename=os.path.join(log_path, "out.log"),
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    level=logging.INFO
)


# # input_dim #需要构建的global名字
# def train_my_model(train_loader, vali_loader, model_flag=0):
#     print("begin to get_model_predict ......")
#     model_path = os.path.join(config.model_path, "Unet_fold_%d.h5" % (model_flag))
#     if os.path.exists(model_path): os.remove(model_path); print("Remove file %s." % (model_path))
#     # base_model = FCN8(config.class_num)
#     base_model = FCN16(config.in_channels, config.class_num)
#     # base_model = FCN8(3)
#     wyl_model = My_Model(config, base_model)
#     if os.path.exists(model_path):
#         wyl_model.model = torch.load(model_path)
#     else:
#         wyl_model.train(train_loader, vali_loader, config.epochs, early_stop=config.early_stop)
#         torch.save(wyl_model.model, model_path)
#     return wyl_model
# 
# 
# def model_training():
#     train_test_split_ratio = 0.8
#     train_vali_split_ratio = 0.7
# 
#     files_root = os.path.join(config.data_path, "img")
#     file_names = os.listdir(files_root)
#     file_names = list(filter(lambda x: ".tif" in x, file_names))
# 
#     # shuffle原文件
#     random.seed(20180122)
#     file_names = random.sample(file_names, len(file_names))
# 
#     trian_file_names = file_names[:int(len(file_names) * train_test_split_ratio)]
#     test_file_names = file_names[int(len(file_names) * train_test_split_ratio):]
# 
#     vali_file_names = trian_file_names[int(len(trian_file_names) * train_vali_split_ratio):]
#     trian_file_names = trian_file_names[:int(len(trian_file_names) * train_vali_split_ratio)]
# 
#     img_files_path = list(map(lambda x: os.path.join(config.data_path, "img", x), test_file_names))
#     gt_files_path = list(map(lambda x: os.path.join(config.data_path, "label", x), test_file_names))
#     save_test_imgs_and_gt(img_files_path, gt_files_path)
# 
#     # 构造训练loader
#     train_set = ImgloaderPostdam_single_channel(
#         config,
#         trian_file_names,
#         return_len=config.train_batch_num,
#        # enhance=True,
#         enhance=False,
#         shuffle=True,
#     )
#     train_loader = DataLoader(
#         train_set,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=config.workers,
#     )
# 
#     # 构造验证loader
#     vali_set = ImgloaderPostdam_single_channel(
#         config,
#         vali_file_names,
#         return_len=config.vali_batch_num,
#         enhance=False,
#         shuffle=False,
#     )
#     vali_loader = DataLoader(
#         vali_set,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=config.workers,
#     )
#     
#     # 构造测试loader
#     test_set = ImgloaderPostdam_single_channel(
#         config,
#         test_file_names,
#         return_len=len(test_file_names),
#         enhance=False,
#         shuffle=False,
#     )
#     test_loader = DataLoader(
#         test_set,
#         batch_size=config.batch_size,
#         shuffle=False,
#         num_workers=config.workers,
#     )
#     
#     ############### 训练模型 #################
#     print("begin to train my model.")
#     wyl_model = train_my_model(train_loader, vali_loader)
#    # prob, F1, overall_F1 = wyl_model.predict(test_loader)
#     prob, result = wyl_model.predict(test_loader)
#     if not os.path.exists('./results'): os.mkdir('./results')
#     result.to_csv("./results/results.csv")
#     # print("pridict label value discript", set(list(np.argmax(prob, axis=1).reshape(-1))))
#     pred_y = np.argmax(prob, axis=1)
#     print(pred_y)
#     save_pred_imgs(pred_y, test_file_names)

def test_loader(files_list):
#     print('[%s] Start loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    start = datetime.now()
    ds = tiledataset(files_list)
    data_loader = DataLoader(
        ds, batch_size=config.batch_size,
#         sampler=data.SequentialSampler(ds),
        num_workers=config.workers)

    prefetcher = tiledata_prefetcher(data_loader, torch.cuda.is_available())
    inputs, targets, index = prefetcher.next()
    k = 0
    while (inputs is not None):
#             inputs = torch.tensor(inputs, dtype=torch.float32)
        img_count = len(inputs)
        inputs, targets, index = prefetcher.next()
        
#         print(k)
#         k += 1
           
    end = datetime.now()
    lapse = end - start
#     print('[%s] End loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    return lapse
def tileimgloadtest(train_files_names):
    # 构造训练loader
    start = datetime.now()
    train_set = TileImgdataset(config, train_files_names,  shuffle=False)   
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
#         shuffle=True,
        num_workers=config.workers,
    )
    k = 0
    # 分批训练样本
    for i, (data_x, data_y) in enumerate(train_loader, 1):
        img_count = len(data_y)

    end = datetime.now()
    lapse = end - start
#     print('[%s] End loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    return lapse

def origimgloadtest(train_files_names):
    # 构造训练loader
    start = datetime.now()
    train_set = OrigImgdataset(config, train_files_names,  shuffle=False)   
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
#         shuffle=True,
        num_workers=config.workers,
    )
    k = 0
    # 分批训练样本
    for i, (data_x, data_y) in enumerate(train_loader, 1):
        img_count = len(data_y)
#         print(k)
#         k+=img_count
#         sample_num += img_count
#         data_x = Variable(data_x.cuda())
#         data_y = Variable(data_y.cuda())
        
#     prefetcher = data_prefetcher(data_loader, args.gpu)
#     with torch.no_grad():
# #         inputs, targets, index = prefetcher.next()
#         inputs, index = prefetcher.next()
#         k = 0
#         while (inputs is not None):
#             inputs = torch.tensor(inputs, dtype=torch.float32)
#             inputs, index = prefetcher.next()
#             k += 1
    end = datetime.now()
    lapse = end - start
#     print('[%s] End loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    return lapse
    

def gen_file_list(geotif):
    file_list = []
    filename = geotif.split('/')[-1]
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, config.BLOCK_SIZE, config.OVERLAP_SIZE)
   
    for xoff, yoff in off_list:    
        file_list.append((filename, xoff, yoff))     
    return file_list


def main():
    tifs_dir = '/data/data/cloud_tif/img'
    tiles_dir = '/data/data/fy4a_tiles'

#     print('[%s] Start loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    st = datetime.now()
    files_offs_list = []
    for root, dirs, files in os.walk(tifs_dir):
        for filename in files:
            if filename.endswith(".tif") and filename.split('_')[1].split('.')[0]=='0330':
                file = os.path.join(root, filename)
                tif_list = gen_file_list(file)
                files_offs_list = files_offs_list+tif_list
    et = datetime.now()
    file = open('/tmp/files_offs_list.txt','w');
    file.write(str(files_offs_list));
    file.close();
    print('the number of file+offs is %s, spend %s' % (len(files_offs_list), (et - st)))
    lapse = origimgloadtest(files_offs_list)
    print('Loading dataset from original image using: %s.' % (lapse))
    
    st = datetime.now()
    tiles = os.listdir(tiles_dir)
    tiles_list = list(filter(lambda x: x.endswith(".tif") and x.split('_')[1]=='0330' , tiles))
    et = datetime.now()
    print('the number of tiles is %s, spend %s' % (len(tiles_list), (et - st)))
#     lapse1 = test_loader(tiles_list)
    lapse1 =tileimgloadtest(tiles_list)
    print('Loading dataset from tiles using: %s.' % (lapse1))
    
if __name__ == '__main__':
    main()
