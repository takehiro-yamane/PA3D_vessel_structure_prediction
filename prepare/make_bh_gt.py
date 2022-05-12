from operator import mod
from random import gauss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


def main(txt_p):
    df = pd.read_csv(txt_p, header=None)
    df = df.astype('int')
    df = df.drop_duplicates()
    # df = df.drop(3, axis=1)
    arr = df.to_numpy()
    npy = np.zeros((1120,1120,400))
    tmp = np.zeros_like(npy)
    index = 1
    for i in tqdm(arr):
        if i[3]==index+1:
            tmp = gaussian_filter(tmp, sigma=1,mode='reflect')
            tmp = tmp/tmp.max()
            tmp *= 255
            npy = np.maximum(npy, tmp)
            index += 1
            tmp = np.zeros_like(npy)
        tmp[i[0],i[1],i[2]]=1

    npy = np.rot90(npy, 3)
    npy = np.fliplr(npy)
    m = np.nanmax(npy,2)
    
    np.save('image/data/bh_gt_sigma1', npy)
    print('image/data/bh_gt')

if __name__ == '__main__':
    n = np.load('image/data/bh_gt_sigma1.npy')
    m = np.nanmax(n,2)
    # cv2.imwrite('image/data/ori_test.tif', m)

    txt_data_path = 'image/data/IPh0009_2on_797_add_20210903_bodyhairs_forYamane.txt'
    main(txt_data_path)