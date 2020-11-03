import os
import netCDF4 as nc
from utils.fy4a import AGRI_L1,read_fy4a_arr
from sklearn import svm
import gdal
from config import config
import numpy as np
import random
from datetime import datetime
import cv2

geo_range = [5, 54.95, 70, 139.95, 0.05] 


fy4afile = os.path.join(config.data_path, 'fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
h08file = os.path.join(config.data_path, 'h08/202007/31/03/NC_H08_20200731_0330_L2CLP010_FLDK.02401_02401.nc')

data = read_fy4a_arr(fy4afile, geo_range)
data = data[:,99:, 300:-99]
data[np.isnan(data)]=0
tmp = np.zeros(data.shape, dtype=np.float32)
data = cv2.normalize(data,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
feat_num = data.shape[0]
xsize = data.shape[1]
ysize = data.shape[2]  

ds = nc.Dataset(h08file, 'r')
true_arr = ds.variables['CLTYPE'][:].data
y_true = true_arr[200:1101, 100:1101]

y_true[y_true == 2] = 255
y_true[y_true == 4] = 255
y_true[y_true == 5] = 255
y_true[y_true == 8] = 255
y_true[y_true == 0] = 255
y_true[y_true == 10] = 255

# n = np.sum(y_true == 255)

x = np.array([])
y = np.array([])

for i in range(xsize):
    for j in range(ysize):
        class_id = y_true[i,j]
        print(class_id)
        if class_id == 255:
            rand = random.random()
            if rand>0.1:
                continue
        else:
            feats = data[:,i,j]
            x=np.append(x, feats)
            y=np.append(y,class_id)
        
x = x.reshape(-1,feat_num)
# X = pd.DataFrame(x)
# Y = pd.DataFrame(y)

print('begin training SVM model......')
clf = svm.SVC()
clf.fit(x, y)
model_npy = '/tmp/svm_%s.npy'%(str(datetime.now()))
np.save(model_npy, Z_KNN)

clf = np.load(model_npy)
print('using SVM model to predict the image.....')
fy4a_pred = '/mnt/win/data/cloud/fy4a_tif/20200413_0330.tif'
ds = gdal.Open(fy4a_pred)
data_x = ds.ReadAsArray()
data_x[np.isnan(data_x)]=0
tmp = np.zeros(data_x.shape, dtype=np.float32)
data_x = cv2.normalize(data_x,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

row = data_x.shape[0]
col = data_x.shape[1]
X = np.array([])
for i in range(col):
    for j in range(row):
        feats = data[:,j,i]
        X=np.append(x, feats)
        
img_SVM = clf.predict(X).reshape(row, col)

dst_path = os.path.join('/tmp/sc_svm01010640.tif')
# save the results into a raster array
with rasterio.Env():
    # Write an array as a raster band to a new 8-bit file. For
    # the new file's profile, we start with the profile of the source
    profile = self.src_profile
    # And then change the band count to 1, set the dtype to uint8, and specify LZW compression.
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw')

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(img_SVM.astype(rasterio.uint8), 1)
    