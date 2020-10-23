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
from modules.models import OrigImgloader
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

# classid2rgb_map = {
#     0: [255, 255, 255],  # "white", 背景
#     1: [255, 0, 0],  # "red",
#     2: [0, 0, 255],  # "blue", 房子
#     3: [0, 255, 255],  # "cyan", 土地
#     4: [0, 255, 0],  # "green", 草地
#     5: [255, 255, 0],  # "yello" 汽车
# }
# 
# 
# def label2rgb(pred_y):
#     # print(set(list(pred_y.reshape(-1))))
#     rgb_img = np.zeros((pred_y.shape[0], pred_y.shape[1], 3))
#     for i in range(len(pred_y)):
#         for j in range(len(pred_y[0])):
#             rgb_img[i][j] = classid2rgb_map.get(pred_y[i][j], [255, 255, 255])
#     return rgb_img.astype(np.uint8)
# 
# 
# # 将矩阵保存成图片
# def save_pred_imgs(test_y, files_name):
#     # print("array to image", test_y.shape)
#     assert test_y.shape[0] == len(files_name), "len(test_files_name) != len(test_y)"
#     if not os.path.exists(config.test_pred_path):
#         os.makedirs(config.test_pred_path)  # 存放模型的地址
#     else:
#         removeDir(config.test_pred_path)
#         os.makedirs(config.test_pred_path)  # 存放模型的地址
#     for i in range(test_y.shape[0]):
#         img = label2rgb(test_y[i])
#         dump_file_name = config.test_pred_path + "/%s" % (files_name[i])
#         # tiff.imsave(dump_file_name, img)
#         cv2.imwrite(dump_file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# #        cv2.imwrite(dump_file_name, img)
# 
# 
# def save_test_imgs_and_gt(imgs_path, gt_path):
#     print(config.test_ori_imgs_dump_path)
#     if not os.path.exists(config.test_ori_imgs_dump_path):
#         os.makedirs(config.test_ori_imgs_dump_path)
#     else:
#         print('not exist')
#         removeDir(config.test_ori_imgs_dump_path)
#         os.makedirs(config.test_ori_imgs_dump_path)  # 存放模型的地址
#     for img_path in imgs_path:
#         file_name = os.path.split(img_path)[-1]
#         moveFileto(img_path, os.path.join(config.test_ori_imgs_dump_path, file_name))
# 
#     if not os.path.exists(config.test_gt_dump_path):
#         os.makedirs(config.test_gt_dump_path)
#     else:
#         removeDir(config.test_gt_dump_path)
#         os.makedirs(config.test_gt_dump_path)  # 存放模型的地址
#     for img_path in gt_path:
#         file_name = os.path.split(img_path)[-1]
#         moveFileto(img_path, os.path.join(config.test_gt_dump_path, file_name))
# 
# 
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
# def get_args():
#     # Get the parameters
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', default='/data/fcn/test-set/')
#     parser.add_argument('--model_path', default='model/')
#     parser.add_argument('--train_f', default='train.lst')
#     parser.add_argument('--val_f', default='val.lst')
#     parser.add_argument('--tbs', type=int, default=8)
#     parser.add_argument('--ag_step', type=int, default=1)
#     parser.add_argument('--tbs_target', type=int, default=8)
# 
#     parser.add_argument('--vbs', type=int, default=32)
#     parser.add_argument('--max_epochs', type=int, default=50)
#     parser.add_argument('--anneal_start', type=int, default=30)
#     parser.add_argument('--save_epoch', type=int, default=0)
#     parser.add_argument('--log_batch', type=int, default=0)
# 
#     parser.add_argument('--optimizer', default='sgd')
#     parser.add_argument('--dtoSGD', type=int, default=False)
#     parser.add_argument('--lookahead', type=int, default=False)
#     parser.add_argument('--la_steps', type=int, default=5)
#     parser.add_argument('--la_alpha', type=float, default=0.8)
#     parser.add_argument('--wd', type=float, default=0)
#     parser.add_argument('--amsgrad', type=int, default=False)
# 
#     parser.add_argument('--lr0', type=float, default=1e-2)
#     parser.add_argument('--warmup')  # constant, gradual
#     parser.add_argument('--warmup_step', type=int)
#     parser.add_argument('--lr1', type=float)
#     parser.add_argument('--mom', type=float, default=0.9)
#     parser.add_argument('--nag', type=int, default=1)
# 
#     parser.add_argument('--n_channels', type=int, default=14)
#     parser.add_argument('--n_filters', type=int, default=64)
#     parser.add_argument('--n_class', type=int, default=6)
#     parser.add_argument('--H', type=int, default=256)
#     parser.add_argument('--W', type=int, default=256)
#     parser.add_argument('--norm_layer', default='none')
#     parser.add_argument('--num_groups', type=int, default=0)
#     parser.add_argument('--group_size', type=int, default=0)
#     parser.add_argument('--in_with_mom', type=int, default=0)
#     parser.add_argument('--in_mom', type=float, default=0.1)
#     parser.add_argument('--affine', type=int, default=1)
#     parser.add_argument('--using_movavg', type=int, default=1)
#     parser.add_argument('--using_bn', type=int, default=1)
#     parser.add_argument('--leps', type=int, default=1)
#     parser.add_argument('--eps', type=float, default=1e-4)
#     parser.add_argument('--wstd', type=int, default=0)
#     parser.add_argument('--act_layer', default='relu')
#     parser.add_argument('--gc', type=int, default=0)
#     parser.add_argument('--gcc', type=int, default=1)
#     parser.add_argument('--gcloc', type=int, default=0)
# 
#     parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))
#     parser.add_argument('--gpu', type=int, default=torch.cuda.is_available())
#     parser.add_argument('--resume')
#     parser.add_argument('--seed', type=int)
#     args = parser.parse_args()
# 
#     return args

def test_loader(files_list,args):
#     print('[%s] Start loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    start = datetime.now()
    ds = dataset(files_list)
    data_loader = data.DataLoader(
        ds, batch_size=config.batch_size,
        sampler=data.SequentialSampler(ds),
        num_workers=config.workers)

    prefetcher = data_prefetcher(data_loader, args.gpu)
    with torch.no_grad():
#         inputs, targets, index = prefetcher.next()
        inputs, index = prefetcher.next()
        k = 0
        while (inputs is not None):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs, index = prefetcher.next()
            print(k)
            k += 1
    end = datetime.now()
    lapse = end - start
#     print('[%s] End loading dataset using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    return lapse


def origimgloadtest(train_files_names):
    # 构造训练loader
    start = datetime.now()
    train_set = OrigImgloader(config, train_files_names,  shuffle=False)   
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
        files = list(filter(lambda x: x.endswith(".tif") and '0330' in x, files))
        for filename in files:
            file = os.path.join(root, filename)
            tif_list = gen_file_list(file)
            files_offs_list = files_offs_list+tif_list
    et = datetime.now()
    
    print('the number of file+offs is %s, spend %s' % (len(files_offs_list), (et - st)))
    lapse = origimgloadtest(files_offs_list)
    print('Loading dataset from original image using: %s.' % (lapse))
    
    st = datetime.now()
    tiles = os.listdir(tiles_dir)
    tiles_list = list(filter(lambda x: x.endswith(".tif") and '0330' in x, tiles))
    et = datetime.now()
    print('the number of tiles is %s, spend %s' % (len(tiles_list), (et - st)))
    lapse1 = test_loader(tiles_list)
    print('Loading dataset from tiles using: %s.' % (lapse1))


if __name__ == '__main__':
    main()
