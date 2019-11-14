from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from MTransform import SyncRandomCrop
import os
from PIL import Image
import numpy as np
import cv2
import torch
from cfg import Configs


class BlurDataSet(Dataset):
    def __init__(self, data_dir, target_dir, aug):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.aug = aug

        if not os.path.exists(data_dir):
            raise RuntimeError("dataset error:", self.target_dir + 'not exists')
        if not os.path.exists(target_dir):
            raise RuntimeError("dataset error:", self.data_dir + 'not exists')

        self.data_name_list = []
        self.target_name_list = []
        for _, _, file_names in os.walk(self.data_dir):
            for fileName in file_names:
                self.data_name_list.append(os.path.join(self.data_dir, fileName))
        for _, _, file_names in os.walk(self.target_dir):
            for fileName in file_names:
                self.target_name_list.append(os.path.join(self.target_dir, fileName))
        data_len = len(self.data_name_list)
        target_len = len(self.target_name_list)
        if data_len != target_len:
            raise RuntimeError("different num of data and target in " + self.data_dir + ' and ' + self.target_dir)
        self.data_name_list.sort()
        self.target_name_list.sort()
        self.size = data_len

        self.aug_transforms = []
        self.aug_transforms.append(transforms.Compose([SyncRandomCrop((256,256)), transforms.RandomHorizontalFlip(1), transforms.RandomRotation((90,90))]))
        self.aug_transforms.append(transforms.Compose([SyncRandomCrop((256,256)), transforms.RandomHorizontalFlip(1), transforms.RandomRotation((180,180))]))
        self.aug_transforms.append(transforms.Compose([SyncRandomCrop((256,256)), transforms.RandomHorizontalFlip(1), transforms.RandomRotation((270,270))]))

        self.input_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        
        self.target_transforms = []
        self.target_transforms.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()]))
        self.target_transforms.append(transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()]))
        self.target_transforms.append(transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()]))
        self.target_transforms.append(transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]))

    def __getitem__(self, item):
        origin_image = Image.open(self.data_name_list[item], "r")
        target_image = Image.open(self.target_name_list[item], 'r')
        
        inputs=[]
        targets=[]

        inputs.append(self.input_transform(origin_image))
        target_images = []
        for tran in self.target_transforms:
            target_images.append((tran(target_image) * 255).long().squeeze())
        targets.append(target_images)

        if self.aug:
            augs_inputs = []
            augs_targets = []
            for aug_trans in self.aug_transforms:
                augs_inputs.append(aug_trans(origin_image))
                aug_trans.transforms[0].rand_fix()
                augs_targets.append(aug_trans(target_image))
                aug_trans.transforms[0].rand_active()

            for i in range(len(augs_inputs)):
                inputs.append(self.input_transform(augs_inputs[i]))
                target_images = []
                for tran in self.target_transforms:
                    target_images.append((tran(augs_targets[i]) * 255).long().squeeze())
                targets.append(target_images)

        return inputs, targets

    def __len__(self):
        return self.size


def test_dataset(data_set):
    size = len(data_set)
    for i in range(size):
        a = data_set.__getitem__(i)
        b = np.transpose(a[0].numpy(), (1, 2, 0))
        c = a[1][0].numpy().astype(np.uint8) * 120
        cv2.imshow('img', b)
        cv2.imshow('mask', c)
        cv2.waitKey(0)


if __name__ == '__main__':
    train_data_set = BlurDataSet("C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\train_image",
                                 "C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\train_gt")
    test_data_set = BlurDataSet("C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_image",
                                "C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_gt")
    loader = DataLoader(train_data_set,batch_size=1,shuffle=False)
    for batch_id, data in enumerate(loader):
        input = torch.cat(data[0],0)
        print(input.shape)
        target = []
        for i in range(4):
            cur=[]
            for t in data[1]:
                cur.append(t[i])
            target.append(torch.cat(cur, 0))

        for i in range(input.shape[0]):
            b = np.transpose(input[i].numpy(), (1, 2, 0))
            c = target[0][i].numpy().astype(np.uint8)
            # d = target[1][i].numpy().astype(np.uint8)
            # e = target[2][i].numpy().astype(np.uint8)
            # f = target[3][i].numpy().astype(np.uint8)
            # c[c == 1] = 0
            # c[c == 2] = 0
            # c[c > 0] = 255
            # d[d == 1] = 0
            # d[d == 2] = 0
            # d[d > 0] = 255
            # e[e == 1] = 0
            # e[e == 2] = 0
            # e[e > 0] = 255
            # f[f == 1] = 0
            # f[f == 2] = 0
            # f[f > 0] = 255
            # cv2.imshow('img', b)
            # cv2.imshow('mask_0', c)
            # cv2.imshow('mask_1', d)
            # cv2.imshow('mask_2', e)
            # cv2.imshow('mask_3', f)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join("C:\\Users\\Whale\\Projects\\result\\gt",str(batch_id)+".png"),c)

        # break
    exit(0)
