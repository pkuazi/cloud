import os
import gdal
import numpy as np
import h5py
import torch
import torch.utils.data as data
import time
import psutil
from datetime import datetime
from config import config
import netCDF4 as nc
import argparse
from utils.fy4a import AGRI_L1,read_fy4a_arr
from utils.gen_tiles_offs import gen_tiles_offs
from utils.epsg2wkt import epsg2wkt
from modules.models import preddataset,predprefetcher
# from prefetcher import data_prefetcher
# from utils.FWIoU import Frequency_Weighted_Intersection_over_Union

BLOCK_SIZE = 256
OVERLAP_SIZE = 0
geo_range = [5, 54.95, 70, 139.95, 0.05] 
minx = geo_range[2]
maxy = geo_range[1]
res = geo_range[4]
 
fy4a_gt = (minx, res,0.0, maxy, 0.0,(-1)*res)
# # 各分辨率文件包含的通道号
# CONTENTS = {'0500M': ('Channel02',),
#             '1000M': ('Channel01', 'Channel02', 'Channel03'),
#             '2000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 8)]),
#             '4000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 15)])}
NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def predict_cpu():
    model_dir = os.path.join(config.model_path, 'model-2020-10-14-21-49-44-213219-best.pth')
    if torch.cuda.is_available():
        net = torch.load(model_dir)
    else:
        net = torch.load(model_dir, map_location=torch.device('cpu'))
    net = net.double()

    # start test
    timestamp = datetime.now()
    print('[%s] Start testing.' % timestamp)
    net.eval()
        
    with torch.no_grad():
        
        fy4afile = os.path.join(config.data_path,'fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731043418_20200731043835_4000M_V0001.HDF')
        filename= fy4afile.split('/')[-1].split('.')[0]
        print('the current fy4afile is %s'%fy4afile)
        fyarr = read_fy4a_arr(fy4afile,geo_range)
         
        xsize, ysize = fyarr.shape[2], fyarr.shape[1]
        off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
         
        pred_arr = np.zeros([ysize, xsize])
        
        for xoff,yoff in off_list:         
            tile_data = fyarr[:,yoff:yoff+BLOCK_SIZE,xoff:xoff+BLOCK_SIZE]
            if np.all(np.isnan(tile_data)) or np.all(tile_data == 255):
                continue
            nanloc = np.isnan(tile_data)
            tile_data[nanloc]=0
        
            tile_data = np.expand_dims(tile_data, axis = 0)
            inputs = torch.from_numpy(tile_data)
    #         outputs = net(inputs.cuda())  # with shape NCHW
            start = time.time()
            print('[%s] Start testing the tile.' % timestamp)
            outputs = net(inputs.cpu())  # with shape NCHW
            _, predict = torch.max(outputs.data, 1)  # with shape NHW
            print('[%s] Finished testing the tile. ' % (time.time()-start))
            tile_pred = predict.cpu().numpy()
            tile_pred = tile_pred[0]
            tile_pred[tile_pred == 0] = 255
            tile_pred[tile_pred == 1] = 6
            tile_pred[tile_pred == 2] = 9
            tile_pred[tile_pred == 3] = 7
            tile_pred[tile_pred == 4] = 1
            tile_pred[tile_pred == 5] = 3
            tile_pred[nanloc[0]] = 255
     
            pred_arr[yoff:yoff+BLOCK_SIZE,xoff:xoff+BLOCK_SIZE]=tile_pred
         
        dst_hdf = os.path.join(config.results_path,'%s_CLT.hdf'%filename)
        #HDF5的写入：    
        f = h5py.File(dst_hdf,'w')   #创建一个h5文件，文件指针是f  
        f['FY4CLT'] = pred_arr                 #将数据写入文件的主键'FY4CLT'下面    
        f.close() 
    print('[%s] Finished testing. ' % datetime.now())
#         tile_pred = predict(tile_data)
#     # compute FWIoU score with H08 data
#         fy = h5py.File('/mnt/win/code/dataservice/cloud/results/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001_CLT.hdf','r')
#         pred_arr = fy['FY4CLT'][()]
#         y_pred = pred_arr[100:,300:1299]
#         y_pred = y_pred.astype('int16')
#         
#         ds = nc.Dataset(os.path.join(config.data_path,'h08/202007/31/04/NC_H08_20200731_0440_L2CLP010_FLDK.02401_02401.nc'), 'r')
#         true_arr = ds.variables['CLTYPE'][:].data
#         y_true = true_arr[200:1100,100:1099]
#         y_true[y_true==2]=255
#         y_true[y_true==4]=255
#         y_true[y_true==5]=255
#         y_true[y_true==8]=255
#         y_true[y_true==0]=255
#         y_true[y_true==10]=255
#         FWIoU = Frequency_Weighted_Intersection_over_Union(y_true, y_pred)
#         print(FWIoU)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))
    parser.add_argument('--gpu', type=int, default=torch.cuda.is_available())
    parser.add_argument('--file_list')
    parser.add_argument('--savename', default='res')
    parser.add_argument('--model', default=os.path.join(config.model_path, 'model-2020-10-14-21-49-44-213219-best.pth'))
    parser.add_argument('--bs', type=int, default=1)

    args = parser.parse_args()
    return args

def fy4a2geotif(fy4afile, dst_file):
    filename= fy4afile.split('/')[-1].split('.')[0]
    print('the current fy4afile is %s'%fy4afile)
    data = read_fy4a_arr(fy4afile,geo_range)
    xsize = data.shape[1]
    ysize = data.shape[2]
    
    fydataType = NP2GDAL_CONVERSION[str(data.dtype)]
    gt = fy4a_gt 
    dst_nbands = data.shape[0]

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, fydataType)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(epsg2wkt('EPSG:4326'))
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(dst_nbands):
            dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    del dst_ds
    return xsize, ysize
    
def gen_file_list(geotif):
    file_list = []
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
   
    for xoff,yoff in off_list:    
        file_list.append((geotif, xoff, yoff))     
    return file_list

# %%
# Test the trained model
def predict(args,files_list):
#     with open(args.file_list, 'r') as f:
#         files = f.read().splitlines()
    ds = preddataset(files_list)
    data_loader = data.DataLoader(
        ds, batch_size=args.bs,
        sampler=data.SequentialSampler(ds),
        num_workers=args.nproc, pin_memory=args.gpu, drop_last=False)
    
    print('[%s] Start test using: %s.' % (datetime.now(), args.model.split('/')[-1]))

    # %%
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    if args.gpu:
        net = torch.load(args.model)
    else:
        net = torch.load(args.model, map_location=torch.device('cpu'))
    print('[%s] Model loaded: %s' % (datetime.now(), args.model))

    # start test
    net.eval()
    correct, total = 0, 0
    H=BLOCK_SIZE
    W=BLOCK_SIZE
    pred = torch.empty((len(ds), H, W), dtype=torch.long)
    labels = torch.empty((len(ds), H, W), dtype=torch.long)
    if args.gpu:
        pred = pred.cuda()
        labels = labels.cuda()
    prefetcher = predprefetcher(data_loader, args.gpu)
    with torch.no_grad():
#         inputs, targets, index = prefetcher.next()
        inputs, index = prefetcher.next()
        k = 0
#         while (inputs is not None) and (targets is not None):
        while (inputs is not None):
#             if args.bs == 1:
#                 sampleID = files[index].split('/')[-1].split('.')[0]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = net(inputs)  # with shape NCHW
            _, predict = torch.max(outputs.data, 1)  # with shape NHW

#             if args.bs == 1:
#                 print('[%5d/%5d]    %s    test_accu: %.3f' % (total, len(files), sampleID, correct_i/H/W))
#             else:
#                 print('[%5d/%5d] test_accu: %.3f' % (total, len(files), correct_i/H/W/targets.shape[0]))
            
            tile_pred = predict[0]
            tile_pred[tile_pred == 0] = 255
            tile_pred[tile_pred == 1] = 6
            tile_pred[tile_pred == 2] = 9
            tile_pred[tile_pred == 3] = 7
            tile_pred[tile_pred == 4] = 1
            tile_pred[tile_pred == 5] = 3
            
            pred[index.tolist()] = tile_pred
#             labels[index.tolist()] = targets

            # prefetch train data
#             inputs, targets, index = prefetcher.next()
            inputs, index = prefetcher.next()
            k += 1

    print('[%s] Finished test.' % datetime.now())

    return pred
    
if __name__ == '__main__':
#     predict_cpu()    
    fy4afile = os.path.join(config.data_path,'fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
    filename = fy4afile.split('/')[-1].split('.')[0]
    fy4a_tif = '/tmp/test1.tif'
    xsize, ysize = fy4a2geotif(fy4afile,fy4a_tif)
    files_list = gen_file_list(fy4a_tif)
    args = get_args()
    pred_list = predict(args,files_list)
    
    pred_arr = np.zeros([xsize,ysize ])
    num = len(files_list)
    for i in range(num):
        _,xoff,yoff = files_list[i]
#         tile_pred = predict.cpu().numpy()
        pred_arr[yoff:yoff+config.BLOCK_SIZE,xoff:xoff+config.BLOCK_SIZE]=pred_list[i].cpu().numpy()
    
    dst_hdf = os.path.join(config.results_path,'%s_CLT2.hdf'%filename)
    #HDF5的写入：    
    f = h5py.File(dst_hdf,'w')   #创建一个h5文件，文件指针是f  
    f['FY4CLT'] = pred_arr                 #将数据写入文件的主键'FY4CLT'下面    
    f.close() 
    
    dst_file = '/tmp/clt2.tif'
    dataType = NP2GDAL_CONVERSION[str(pred_arr.dtype)]
    gt = fy4a_gt 

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, 1, dataType)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(epsg2wkt('EPSG:4326'))
    dst_ds.GetRasterBand(1).WriteArray(pred_arr)
    
    



    
    
