from Net import Net, Net_Bn
from cfg import Configs
from DataSetLoader.MDataSet import BlurDataSet
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import os


if __name__ == '__main__':
    n = Net_Bn()
    n.load_model(Configs["pre_path"],Configs["model_save_path"])
    exit(0)