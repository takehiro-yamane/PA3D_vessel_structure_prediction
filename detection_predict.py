from datetime import datetime
from PIL import Image
from numpy.lib.index_tricks import OGridClass
from numpy.lib.type_check import _imag_dispatcher, real_if_close
import torch
import torch.utils.data
import numpy as np
from pathlib import Path
from utils import CellImageLoad, ConfirmImageLoad
from collections import OrderedDict

import cv2
# from networks import UNet
from networks import VNet
from utils import local_maxima, show_res, optimum, target_peaks_gen, remove_outside_plot
import argparse
from tqdm import tqdm

predict = '0625'

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/detection",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./weight/10000.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=16, type=int
    )
    args = parser.parse_args()
    return args


class Predict:
    def __init__(self, args):
        self.net = args.net
        self.gpu = args.gpu

        self.ori_path = args.input_path

        self.save_ori_path = args.output_path / Path("ori")
        self.save_pred_path = args.output_path / Path("pred")

        self.save_ori_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_path.mkdir(parents=True, exist_ok=True)

    def pred(self, ori):
        img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1], ori.shape[2])
        )

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
            mask_pred = self.net(img)
        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    def main(self):
        self.net.eval()
        ori = np.load('/home/kazuya/WSISPDR_vnet/image/test/ori/1_ON_20170228-164439_IPh006_R_PA756.npy')
        pre_img = self.pred(ori)
        # cv2.imwrite(str(self.save_pred_path / Path("%05d.tif" % i)), pre_img)
        # cv2.imwrite(str(self.save_ori_path / Path("%05d.tif" % i)), ori)
        # np.save('/home/kazuya/dataset/experiment_vnet/conf/confirm2/ori', ori)
        np.save('/home/kazuya/dataset/experiment_vnet/conf/confirm2/pre', pre_img)
        savedir = Path('/home/kazuya/dataset/experiment_vnet/conf/confirm2/pre-0205-006')
        savedir.mkdir()
        for i in range(pre_img.shape[2]):
            cv2.imwrite('{}/{:04}.tif'.format(savedir, i), pre_img[:,:,i])



class PredictNet(Predict):
    def convert(self, image):
        tmp = image[0,:,:,:]
        max0 = np.nanmax(tmp, axis=0)
        max1 = np.nanmax(tmp, axis=1)
        max2 = np.nanmax(tmp, axis=2)
        #max0 = max0/(np.amax(max0)-np.amin(max0))
        max0 *= 255
        #max1 = max1/(np.amax(max1)-np.amin(max1))
        max1 *= 255
        #max2 = max2/(np.amax(max2)-np.amin(max2))
        max2 *= 255

        return max0, max1, max2

    def convert_max(self, image):
        tmp = image[0,:,:,:]
        max0 = np.nanmax(tmp, axis=0)
        max1 = np.nanmax(tmp, axis=1)
        max2 = np.nanmax(tmp, axis=2)
        max0 = np.where(max0.max()*0.7 > max0, max0*0.5, max0)
        max0 = max0/(np.amax(max0))
        max0 *= 255
        max1 = np.where(max1.max()*0.7 > max1, max1*0.5, max1)
        max1 = max1/(np.amax(max1))
        max1 *= 255
        max2 = np.where(max2.max()*0.7 > max2, max2*0.5, max2)
        max2 = max2/(np.amax(max2))
        max2 *= 255

        return max0, max1, max2

    def pred(self, ori):
        # img = (ori.astype(np.float32) / ori.max()).reshape(
        #     (1, ori.shape[0], ori.shape[1], ori.shape[2])
        # )
        ori = ori.unsqueeze(0)
        with torch.no_grad():
            # img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                ori = ori.cuda()
            mask_pred = self.net(ori)
        # pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        pre_img = mask_pred.detach().cpu().numpy()
        #pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    def main(self):
        self.net.eval()
        # path def
        #paths = sorted(self.ori_path.glob("*.tif"))


        ori = './image/test/ori/1_ON_20170228-164439_IPh006_R_PA756.npy'
        data_loader = self.predict_data(ori)
        
        for i, data in enumerate(tqdm(data_loader)):
            imgs = data
            # ori = np.array(Image.open(path))
            # pre_img = self.pred(ori)
            pre_img = self.pred(imgs)
            pre_img = pre_img[0,0,:,:,:]
            imgs = imgs[0,:,:,:]
            imgs = imgs.detach().cpu().numpy()
            pre_img *= 255
            imgs *= 255


            if i%9==0:
                tmp_imgs = imgs
                tmp_pre = pre_img
            elif i%9==8:
                tmp_imgs = np.concatenate([tmp_imgs, imgs[:, 32:, :]], 1)
                tmp_pre = np.concatenate([tmp_pre, imgs[:, 32:, :]], 1)
            else:
                tmp_imgs = np.concatenate([tmp_imgs, imgs], 1)
                tmp_pre = np.concatenate([tmp_pre, pre_img], 1)

            if (i+1)%9==0:
                if (i+1)/9==1 or (i+1)/9==10 or (i+1)/9==19 or (i+1)/9==28 or (i+1)/9==37 or (i+1)/9==46 or (i+1)/9==55:
                    pre_arr = tmp_pre
                    img_arr = tmp_imgs
                elif (i+1)%81==0:
                    if i==80:
                        pre3d = np.concatenate([pre_arr, tmp_pre[32:,:,:]], 0)
                        img3d = np.concatenate([img_arr, tmp_imgs[32:,:,:]], 0)
                    elif i==566:
                        pre_arr = np.concatenate([pre_arr, tmp_pre[32:,:,:]], 0)
                        img_arr = np.concatenate([img_arr, tmp_imgs[32:,:,:]], 0)
                        pre3d = np.concatenate([pre3d, pre_arr[:,:,48:]], 2)
                        img3d = np.concatenate([img3d, img_arr[:,:,48:]], 2)
                    else:
                        pre_arr = np.concatenate([pre_arr, tmp_pre[32:,:,:]], 0)
                        img_arr = np.concatenate([img_arr, tmp_imgs[32:,:,:]], 0)
                        pre3d = np.concatenate([pre3d, pre_arr], 2)
                        img3d = np.concatenate([img3d, img_arr], 2)
                else:
                    pre_arr = np.concatenate([pre_arr, tmp_pre], 0)
                    img_arr = np.concatenate([img_arr, tmp_imgs], 0)


            # if (i+1)%8==0:
            #     if i==7 or i==64+7 or i==64+64+7:
            #         pre_arr = tmp_pre
            #         img_arr = tmp_imgs
            #     else:
            #         pre_arr = np.concatenate([pre_arr, tmp_pre], 0)
            #         img_arr = np.concatenate([img_arr, tmp_imgs], 0)
            # if i%64==63:
            #     if i==63:
            #         img_3d = img_arr
            #         pre_3d = pre_arr
            #     else:
            #         img_3d = np.concatenate([img_3d, img_arr], 2)
            #         pre_3d = np.concatenate([pre_3d, pre_arr], 2)
            #     img_arr = np.empty((1024, 1024, 64))
            #     pre_arr = np.empty((1024, 1024, 64))
                    
        # img_3d = img_3d.astype(np.uint8)
        # pre_3d = pre_3d.astype(np.uint8)
        # img_3d = img3d.astype(np.uint8)
        pre_3d = pre3d.astype(np.uint8)
        # np.save('/home/kazuya/dataset/experiment_vnet/tiff_file/pred/ori', img_3d)
        np.save('/home/kazuya/dataset/experiment_vnet/tiff_file/pred/pre', pre_3d)
        # self.conv_kurumi('/home/kazuya/dataset/experiment_vnet/tiff_file/pred/original', img_3d)
        # self.conv_kurumi('/home/kazuya/dataset/experiment_vnet/tiff_file/pred/prediction', pre_3d)
        savedir = Path('/home/kazuya/dataset/experiment_vnet/conf/confirm2/pre-0205-006')
        savedir.mkdir(exist_ok=True)
        for i in range(pre_3d.shape[2]):
            cv2.imwrite('{}/{:04}.tif'.format(savedir, i), pre_3d[:,:,i])


    def predict_data(self, path):
        img = np.load(path)
        img = img / img.max()
        front, back, top, bottom, left, right = self.crop_param(img.shape)
        datas=[]
        
        tmp_f = front
        tmp_b = back

        for i in range(6):
            tmp_t = top
            tmp_bo = bottom
        
            for j in range(8):
                if j!=0:
                    tmp_t += 128
                    tmp_bo += 128
                tmp_l = left
                tmp_r = right
                
                for k in range(8):    
                    tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
                    tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
                    datas.append(tmp_img.unsqueeze(0))
                    tmp_l += 128
                    tmp_r += 128

                ##conplement edge
                tmp_l -= 32
                tmp_r -= 32
                tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
                tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
                datas.append(tmp_img.unsqueeze(0))

            tmp_t += 96
            tmp_bo += 96
            tmp_l = left
            tmp_r = right
            for l in range(8):
                tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
                tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
                datas.append(tmp_img.unsqueeze(0))
                tmp_l += 128
                tmp_r += 128
            tmp_l -= 32
            tmp_r -= 32
            tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
            tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
            datas.append(tmp_img.unsqueeze(0))


            tmp_f += 64
            tmp_b += 64

        tmp_f -= 48
        tmp_b -= 48
        tmp_t = top
        tmp_bo = bottom
        for j in range(8):
            if j!=0:
                tmp_t += 128
                tmp_bo += 128
            tmp_l = left
            tmp_r = right
            
            for k in range(8):    
                tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
                tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
                datas.append(tmp_img.unsqueeze(0))
                tmp_l += 128
                tmp_r += 128

            ##conplement edge
            tmp_l -= 32
            tmp_r -= 32
            tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
            tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
            datas.append(tmp_img.unsqueeze(0))

        tmp_t += 96
        tmp_bo += 96
        tmp_l = left
        tmp_r = right
        for l in range(8):
            tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
            tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
            datas.append(tmp_img.unsqueeze(0))
            tmp_l += 128
            tmp_r += 128
        tmp_l -= 32
        tmp_r -= 32
        tmp_img = img[tmp_t:tmp_bo, tmp_l:tmp_r, tmp_f:tmp_b]
        tmp_img = torch.from_numpy(tmp_img.astype(np.float32))
        datas.append(tmp_img.unsqueeze(0))

        return datas

    def crop_param(self, shape):
        crop_size=(128, 128, 64)
        d, h, w = shape
        top = 0
        left = 0
        front = 0
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        back = front + crop_size[2]
        return front, back, top, bottom, left, right

    def conv_kurumi(self, savepath, npy):
        # npy=npy/npy.max()
        # npy*=255
        npy = npy.transpose(1,2,0)
        npy = npy.transpose(1,2,0)
        x, y, z=npy.shape

        for i in tqdm(range(x)):
            Path(savepath).mkdir(exist_ok=True)
            cv2.imwrite("{}/{:04}.tif".format(savepath, i), npy[i].astype(np.uint8))



def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

class PredictFmeasure(Predict):
    def __init__(self, args):
        super().__init__(args)
        self.ori_path = args.input_path / Path("ori")
        self.gt_path = args.input_path / Path("gt")

        self.save_gt_path = args.output_path / Path("gt")
        self.save_error_path = args.output_path / Path("error")
        self.save_txt_path = args.output_path / Path("f-measure.txt")

        self.save_gt_path.mkdir(parents=True, exist_ok=True)
        self.save_error_path.mkdir(parents=True, exist_ok=True)

        self.peak_thresh = 100
        self.dist_peak = 2
        self.dist_threshold = 10

        self.tps = 0
        self.fps = 0
        self.fns = 0

    def cal_tp_fp_fn(self, ori, gt_img, pre_img, i):
        gt = target_peaks_gen((gt_img).astype(np.uint8))
        res = local_maxima(pre_img, self.peak_thresh, self.dist_peak)
        associate_id = optimum(gt, res, self.dist_threshold)

        gt_final, no_detected_id = remove_outside_plot(
            gt, associate_id, 0, pre_img.shape
        )
        res_final, overdetection_id = remove_outside_plot(
            res, associate_id, 1, pre_img.shape
        )

        show_res(
            ori,
            gt,
            res,
            no_detected_id,
            overdetection_id,
            path=str(self.save_error_path / Path("%05d.tif" % i)),
        )
        cv2.imwrite(str(self.save_pred_path / Path("%05d.tif" % (i))), pre_img)
        cv2.imwrite(str(self.save_ori_path / Path("%05d.tif" % (i))), ori)
        cv2.imwrite(str(self.save_gt_path / Path("%05d.tif" % (i))), gt_img)

        tp = associate_id.shape[0]
        fn = gt_final.shape[0] - associate_id.shape[0]
        fp = res_final.shape[0] - associate_id.shape[0]
        self.tps += tp
        self.fns += fn
        self.fps += fp

    def main(self):
        self.net.eval()
        # path def
        path_x = sorted(self.ori_path.glob("*.tif"))
        path_y = sorted(self.gt_path.glob("*.tif"))

        z = zip(path_x, path_y)

        for i, b in enumerate(z):
            import gc

            gc.collect()
            ori = cv2.imread(str(b[0]), 0)[:512, :512]
            gt_img = cv2.imread(str(b[1]), 0)[:512, :512]

            pre_img = self.pred(ori)

            self.cal_tp_fp_fn(ori, gt_img, pre_img, i)
        if self.tps == 0:
            f_measure = 0
        else:
            recall = self.tps / (self.tps + self.fns)
            precision = self.tps / (self.tps + self.fps)
            f_measure = (2 * recall * precision) / (recall + precision)

        print(precision, recall, f_measure)
        with self.save_txt_path.open(mode="a") as f:
            f.write("%f,%f,%f\n" % (precision, recall, f_measure))


if __name__ == "__main__":
    args = parse_args()

    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)

    # net = UNet(n_channels=1, n_classes=1)
    net = VNet()

    weight = torch.load(args.weight_path, map_location="cpu")
    weight = fix_model_state_dict(weight)
    # net.load_state_dict(torch.load(args.weight_path, map_location="cpu"))
    net.load_state_dict(weight)

    args.gpu=True

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)

    args.net = net

    pred = PredictNet(args)
    # pred = Predict(args)

    pred.main()


