import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate
import torch.utils.data



def local_maxima(img, threshold=100, dist=2):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data


class PA3DImageLoad(object):
    def __init__(self, ori_path, gt1_path, gt2_path, crop_size=(128, 128, 64)):
        self.ori_paths = ori_path
        self.gt1_paths = gt1_path
        self.gt2_paths = gt2_path
        self.crop_size = crop_size

        img = np.load(str(ori_path[0]))
        self.img = img / img.max()

        gt1 = np.load(str(gt1_path[0]))
        self.gt1 = gt1 / gt1.max()

        gt2 = np.load(str(gt2_path[0]))
        self.gt2 = gt2 / gt2.max()


    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        d, h, w = shape
        front = random.randint(0, d - self.crop_size[0])
        top = random.randint(0, h - self.crop_size[1])
        left = random.randint(0, w - self.crop_size[2])
        back = front + self.crop_size[0]
        bottom = top + self.crop_size[1]
        right = left + self.crop_size[2]
        return front, back, top, bottom, left, right

    def __getitem__(self, data_id):

        flg=0

        #少なくとも１つのgtの値がすべて0とならないようにクロップする．
        while(flg==0):
            front, back, top, bottom, left, right = self.random_crop_param(self.img.shape)
            img = self.img[front:back, top:bottom, left:right]
            gt1 = self.gt1[front:back, top:bottom, left:right]
            gt2 = self.gt2[front:back, top:bottom, left:right]
            if gt1.max() > 0.1 or gt2.max() > 0.1:
                flg = 1
                break
        
        rand_value = random.randint(0, 3)
        img = np.rot90(img, rand_value)
        gt1 = np.rot90(gt1, rand_value)
        gt2 = np.rot90(gt2, rand_value)

        rand_value = random.randint(0, 1)
        rand_axis = random.randint(0, 2)
        if rand_value == 1:
            img = np.flip(img, axis = rand_axis)
            gt1 = np.flip(gt1, axis = rand_axis)
            gt2 = np.flip(gt2, axis = rand_axis)

        img = torch.from_numpy(img.astype(np.float32))
        gt1 = torch.from_numpy(gt1.astype(np.float32))
        gt2 = torch.from_numpy(gt2.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt1": gt1.unsqueeze(0), "gt2": gt2.unsqueeze(0)}

        return datas


if __name__ == "__main__":
    for i in range(10):
        a = random.randint(0,2)
        print(a)
    print('test')