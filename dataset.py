import tifffile,gdal
import numpy as np
import torch
import torch.utils.data as data
from config import config 

class tiledataset(data.Dataset):
    def __init__(self, file_names=None):
        self.file_names = file_names
#         self.img = [x.replace('/gt/', '/imgs/') for x in self.gt]

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
        return data_x, data_y ,index
    
#         input = torch.from_numpy(np.transpose(fy4a_tile_data, (2, 0, 1)))  # HWC to CHW

#         target = tifffile.imread(self.gt[index])
#         # if len(target.shape) == 2:
#         #     target = np.expand_dims(target, axis=0)  # HW to CHW
#         # target -= 1
#         h, w = target.shape
#         label = np.zeros((h, w), dtype=np.int8)
#         label[target == 6] = 1
#         label[target == 9] = 2
#         label[target == 7] = 3
#         label[target == 1] = 4
#         label[target == 3] = 5
#         label = torch.as_tensor(label, dtype=torch.long)
# 
#         fy_bands = tifffile.imread(self.img[index]).astype('float32')
#         fy_bands[np.isnan(fy_bands)] = 255
#         input = torch.from_numpy(np.transpose(fy_bands, (2, 0, 1)))  # HWC to CHW


    def __len__(self):
        return len(self.img)

    def filelist(self):
        return self.img

class dataset(data.Dataset):
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
#         input = torch.from_numpy(np.transpose(fy4a_tile_data, (2, 0, 1)))  # HWC to CHW

#         target = tifffile.imread(self.gt[index])
#         # if len(target.shape) == 2:
#         #     target = np.expand_dims(target, axis=0)  # HW to CHW
#         # target -= 1
#         h, w = target.shape
#         label = np.zeros((h, w), dtype=np.int8)
#         label[target == 6] = 1
#         label[target == 9] = 2
#         label[target == 7] = 3
#         label[target == 1] = 4
#         label[target == 3] = 5
#         label = torch.as_tensor(label, dtype=torch.long)
# 
#         fy_bands = tifffile.imread(self.img[index]).astype('float32')
#         fy_bands[np.isnan(fy_bands)] = 255
#         input = torch.from_numpy(np.transpose(fy_bands, (2, 0, 1)))  # HWC to CHW

        return input, index

    def __len__(self):
        return len(self.img)

    def filelist(self):
        return self.img
