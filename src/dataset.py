import tifffile
import numpy as np
import torch
import torch.utils.data as data


class dataset(data.Dataset):
    def __init__(self, file_path=None):
        super(dataset, self).__init__()
        self.gt = file_path
        self.img = [x.replace('/gt/', '/imgs/') for x in self.gt]

    def __getitem__(self, index):
        target = tifffile.imread(self.gt[index])
        # if len(target.shape) == 2:
        #     target = np.expand_dims(target, axis=0)  # HW to CHW
        # target -= 1
        h, w = target.shape
        label = np.zeros((h, w), dtype=np.int8)
        label[target == 6] = 1
        label[target == 9] = 2
        label[target == 7] = 3
        label[target == 1] = 4
        label[target == 3] = 5
        label = torch.as_tensor(label, dtype=torch.long)

        fy_bands = tifffile.imread(self.img[index]).astype('float32')
        fy_bands[np.isnan(fy_bands)] = 255
        input = torch.from_numpy(np.transpose(fy_bands, (2, 0, 1)))  # HWC to CHW

        return input, label, index

    def __len__(self):
        return len(self.gt)

    def filelist(self):
        return self.gt, self.img
