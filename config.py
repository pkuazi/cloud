# -*- coding:utf-8 -*-
import os
from easydict import EasyDict as edict
import psutil
if not os.path.exists("./logs/"): os.makedirs("./logs/")
if not os.path.exists("./cache/model/"): os.makedirs("./cache/model/")  # 存放模型的地址

config = edict()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config.BLOCK_SIZE = 224
config.OVERLAP_SIZE = 0
# config.data_path = os.path.join(BASE_DIR,"data")
config.data_path = '/data/data'
config.img_path = '/data/data/cloud_tif/img'
config.gt_path = '/data/data/cloud_tif/gt'

config.tile_img_path = '/data/data/fy4a_tiles'
config.tile_gt_path = '/data/data/h08_tiles'

config.test_path='/data/tz2020/ship_detection/test/src'

config.train_imgs_path = os.path.join(BASE_DIR,"data/train-set")
config.test_imgs_path = os.path.join(BASE_DIR,"data/test-set/imgs")

config.test_pred_path = os.path.join(BASE_DIR,"data/test-set/pred")
config.test_gt_dump_path = os.path.join(BASE_DIR,"data/test-set/gt")
config.test_ori_imgs_dump_path = os.path.join(BASE_DIR,"data/test-set/imgs")

config.model_path = os.path.join(BASE_DIR,"model/")
config.results_path = os.path.join('/tmp','results/')
config.in_channels=14
config.class_num = 6
config.img_rows = 224
config.img_cols = 224
# config.img_rows = 800
# config.img_cols = 800
config.train_batch_num = 100 # 1个epoch有train_batch_num个样本
config.vali_batch_num = 50 # 1个epoch有train_batch_num个样本
config.batch_size = 8  # 深度模型 分批训练的批量大小
config.epochs = 150  # 总共训练的轮数（实际不会超过该轮次，因为有early_stop限制）
config.early_stop = 30  # 最优epoch的置信epochs
config.folds = 5 # 使用5折交叉验证
config.workers = psutil.cpu_count(logical=True)



