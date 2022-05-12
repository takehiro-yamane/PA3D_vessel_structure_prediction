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


class CellImageLoad(object):
    def __init__(self, ori_path, gt_path, mask_path, crop_size=(128, 128, 64)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path
        self.crop_size = crop_size



        img = np.load(str(ori_path[0]))
        self.img = img / img.max()

        gt_name = self.gt_paths[0]
        gt = cv2.imread(str(gt_name), 0)
        gt = np.load(str(gt_name))
        self.gt = gt / gt.max()

        mask_name = self.mask_paths[0]
        self.mask = np.load(str(mask_name))
        #self.mask.transpose(0, 2, 1)

    def __len__(self):
        return len(self.ori_paths)
        # return self.batch_size

    def random_crop_param(self, shape):
        d, h, w = shape
        # front = np.random.randint(200, d - self.crop_size[0]-200)
        # top = np.random.randint(200, h - self.crop_size[1])
        # left = np.random.randint(200, w - self.crop_size[2]-200)
        # front = random.randint(200, d - self.crop_size[0]-200)
        front = random.randint(205, 981 - self.crop_size[0])
        # top = random.randint(200, h - self.crop_size[1])
        top = random.randint(276, 763 - self.crop_size[1])
        # left = random.randint(200, w - self.crop_size[2]-200)
        left = random.randint(220, w - self.crop_size[2])
        # front = 550
        # top = 240
        # left = 480
        back = front + self.crop_size[0]
        bottom = top + self.crop_size[1]
        right = left + self.crop_size[2]
        return front, back, top, bottom, left, right

    def __getitem__(self, data_id):

        flg=0

        while(flg==0):
                
            # img_name = self.ori_paths[data_id]
            # # #img = cv2.imread(str(img_name), 0)[:880]
            # img = np.load(str(img_name))
            # img = img / img.max()

            # gt_name = self.gt_paths[data_id]
            # # #gt = cv2.imread(str(gt_name), 0)
            # gt = np.load(str(gt_name))
            # gt = gt / gt.max()


            # mask_name = self.mask_paths[data_id]
            # mask = np.load(str(mask_name))

            # data augumentation
            front, back, top, bottom, left, right = self.random_crop_param(self.img.shape)
            # front, back, top, bottom, left, right = self.random_crop_param(img.shape)

            img = self.img[front:back, top:bottom, left:right]
            gt = self.gt[front:back, top:bottom, left:right]
            mask = self.mask[front:back, top:bottom, left:right]

            if gt.max() > 0.1:
                flg = 1
                break
        # img = img[front:back, top:bottom, left:right]
        # gt = gt[front:back, top:bottom, left:right]
        # mask = mask[front:back, top:bottom, left:right]

        rand_value = random.randint(0, 3)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value)
        mask = rotate(mask, 90 * rand_value)

        rand_value = random.randint(0, 1)
        rand_axis = random.randint(0, 2)
        if rand_value == 1:
            img = np.flip(img, axis = rand_axis)
            gt = np.flip(gt, axis = rand_axis)
            mask = np.flip(mask, axis = rand_axis)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))


        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "mask": mask.unsqueeze(0)}

        return datas


class ConfirmImageLoad(object):
    def __init__(self, ori_path, gt_path, mask_path, crop_size=(128, 128, 64)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path
        self.crop_size = crop_size

        img = np.load(str(ori_path[0]))
        self.img = img / img.max()

        gt_name = self.gt_paths[0]
        #gt = cv2.imread(str(gt_name), 0)
        gt = np.load(str(gt_name))
        self.gt = gt / gt.max()

        mask_name = self.mask_paths[0]
        self.mask = np.load(str(mask_name))

    def __len__(self):
        return len(self.ori_paths)

    def crop_param(self, shape):
        d, h, w = shape
        front = 480
        top = 300
        left = 200
        back = front + self.crop_size[0]
        bottom = top + self.crop_size[1]
        right = left + self.crop_size[2]
        return front, back, top, bottom, left, right

    def __getitem__(self, data_id):
        
        front, back, top, bottom, left, right = self.crop_param(self.img.shape)

        img = self.img[front:back, top:bottom, left:right]
        gt = self.gt[front:back, top:bottom, left:right]
        mask = self.mask[front:back, top:bottom, left:right]

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "mask": mask.unsqueeze(0)}

        return datas





if __name__ == "__main__":

    def predict_data(ori_path):
        img = np.load(str(ori_path))
        img = img / img.max()
        front, back, top, bottom, left, right = crop_param(img.shape)
        datas=[]
        
        tmp_l = left
        tmp_r = right
        tmp_f = front
        tmp_b = back
        for i in range(7):
            tmp_l = left
            tmp_r = right
            tmp_f += 64
            tmp_b += 64
            for j in range(10):
                tmp_img = img[front:back, tmp_t:tmp_b, tmp_l:tmp_r]
                tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
                datas.append(tmp_img.unsqueeze(0))
                tmp_l += 32 
                tmp_r += 32

        return datas

    def crop_param(shape):
        crop_size=(128, 128, 64)
        d, h, w = shape
        front = 300
        top = 230
        left = 300
        back = front + crop_size[0]
        bottom = top + crop_size[1]
        right = left + crop_size[2]
        return front, back, top, bottom, left, right

    ori_path = '/home/kazuya/WSISPDR/image/test/ori/3Dvessel_coordinate_006_trans.npy'
    data_loader = predict_data(ori_path)
    # predict_dataset_loader = torch.utils.data.DataLoader(
    #     data_loader, batch_size=16, shuffle=True, num_workers=0)

    for i, data in enumerate(data_loader):
        imgs = data

    
    print('finish')
