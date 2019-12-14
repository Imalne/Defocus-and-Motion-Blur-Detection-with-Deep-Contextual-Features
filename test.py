
from DataSetLoader.MDataSet import BlurDataSet
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import os
from models.fpn import FPN
from models.resnet import ResNet34,ResNet50
from cfg import Configs

if __name__ == '__main__':
    dataset = BlurDataSet(Configs["test_image_dir"],Configs["test_mask_dir"],True)

    for i,data in enumerate(dataset):
        image,targets = data
        img = np.transpose(image.numpy(),(1,2,0))
        cv2.imshow("img",img)
        target = np.transpose(targets[0].numpy(),(1,2,0))*120
        cv2.imshow("target",target)
        cv2.waitKey(0)
    exit(0)