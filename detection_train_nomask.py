from tqdm import tqdm
from torch import optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torch.nn as nn
import torch.nn
# from utils import CellImageLoad, ConfirmImageLoad
from utils.load_nomask import PA3DImageLoad
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from networks import UNet
from networks import VNet
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="./image/train",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="./image/val",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", default=True, help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=24, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=500, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


class _TrainBase:
    def __init__(self, args):
        self.writer = SummaryWriter()
        ori_paths = self.gather_path(args.train_path, "ori")*args.batch_size
        gt_bh_paths = self.gather_path(args.train_path, "gt_bh")*args.batch_size
        gt_vessel_paths = self.gather_path(args.train_path, "gt_vessel")*args.batch_size
        data_loader = PA3DImageLoad(ori_paths, gt_bh_paths, gt_vessel_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=24
        )
        self.number_of_traindata = data_loader.__len__()

        # ori_paths = self.gather_path(args.val_path, "ori")
        # gt_paths = self.gather_path(args.val_path, "gt")
        # data_loader = CellImageLoad(ori_paths, gt_paths, mask_path)
        # data_loader = ConfirmImageLoad(ori_paths, gt_paths)
        # self.val_loader = torch.utils.data.DataLoader(
        #     data_loader, batch_size=5, shuffle=False, num_workers=24
        # )

        
        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = args.net

        self.train = None
        self.val = None
        self.N_train = None
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning_rate)

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
        return ori_paths


    def maskedmseloss(self, input, target):
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                    "This will likely lead to incorrect results due to broadcasting. "
                    "Please ensure they have the same size.".format(target.size(), input.size()),
                    stacklevel=2)
        ret = (input - target) ** 2
        ret_gt = ret[target > 0.1]
        ret = torch.mean(ret)
        ret_gt = torch.mean(ret_gt)
        ret = ret + ret_gt 
        return ret


class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat):
        return self.criterion(masks_probs_flat, true_masks_flat)
        # return self.maskedmseloss(masks_probs_flat, true_masks_flat, epoch)

    def main(self):
        iteration = 0
        for epoch in tqdm(range(self.epochs)):
            str_a = "Starting epoch {}/{}.".format(epoch + 1, self.epochs)
            print(str_a+"\n"+"\033[2A",end="")

            self.net.train()
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks_bh = data["gt1"]
                true_masks_vessel = data["gt2"]
                
                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks_bh = true_masks_bh.cuda()
                    true_masks_vessel = true_masks_vessel.cuda()

                masks_pred_bh, masks_pred_vessel = self.net(imgs)
                loss_bh = self.loss_calculate(masks_pred_bh, true_masks_bh, epoch)
                loss_vessel = self.loss_calculate(masks_pred_vessel, true_masks_vessel, epoch)
                self.writer.add_scalar('loss_bodyhair', loss_bh, epoch)
                self.writer.add_scalar('loss_vessel', loss_vessel, epoch)

                loss = loss_bh+loss_vessel
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            iteration += 1
        torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.writer.close()


def main(train_path, save_weight_path, epochs):
    args = parse_args()
    args.gpu = True
    args.train_path = [Path(train_path)]
    # args.val_path = [Path(args.val_path)]
    # save weight path
    args.weight_path = Path(save_weight_path)
    args.epochs=epochs

    # define model
    net = VNet()
    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)
    args.net = net
    train = TrainNet(args)
    train.main()

if __name__ == "__main__":
    train_p = 'image/train'
    save_w_p = 'Result/0511'
    main(train_p, save_w_p, 20000)