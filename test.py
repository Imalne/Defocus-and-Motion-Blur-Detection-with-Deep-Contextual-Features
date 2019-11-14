from Net import Net
from cfg import Configs
from DataSetLoader.MDataSet import BlurDataSet
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import os


if __name__ == '__main__':
    out_image_dir = "C:\\Users\\Whale\\Documents\\DataSets\\CUHK_aug\\train_aug_image"
    out_gt_dir = "C:\\Users\\Whale\\Documents\\DataSets\\CUHK_aug\\train_aug_gt"
    test_dataset = BlurDataSet(data_dir=Configs["train_image_dir"], target_dir=Configs["train_mask_dir"],aug=Configs["augmentation"])
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    count = 0
    for _, (inps, tars) in enumerate(loader):
        images = []
        for inp in inps:
            images.append(np.transpose(inp[0].numpy(), (1, 2, 0)))
        targets = []
        for tar in tars:
            targets.append((tar[0].numpy().astype(np.uint8)).squeeze(0))
        num = len(images)

        for i in range(num):
            cv2.imshow("image" + '_' + str(i), images[i])
            cv2.imshow("target" + '_' + str(i), targets[i]*127)
            # cv2.imwrite(os.path.join(out_image_dir,str.format("{:5d}.png",count)),images[i]*255)
            # cv2.imwrite(os.path.join(out_gt_dir,str.format("{:5d}.png",count)),targets[i])
            # print(count)
            # count += 1
        cv2.waitKey(0)
    exit(0)