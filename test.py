
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
    encoder = ResNet34().cuda()
    n = FPN.fromConfig(Configs).cuda()
    n(torch.rand((1,3,224,224)).cuda())
    exit(0)