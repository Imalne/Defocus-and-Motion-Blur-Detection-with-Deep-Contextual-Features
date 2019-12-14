from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from MTransform import SyncRandomCrop
import os
from PIL import Image
import numpy as np
import cv2
import torch
from cfg import Configs
import albumentations as albu


def get_transforms(size: int):
    pipeline = albu.Compose([albu.RandomCrop(size, size, always_apply=True),albu.VerticalFlip(),albu.RandomRotate90(always_apply=True)], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


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
        self.transform = get_transforms(224)
        self.multi_scale_transform = [albu.Resize(224, 224),albu.Resize(112, 112)]

    def __getitem__(self, item):
        image = np.array(Image.open(self.data_name_list[item],'r'))
        target = np.array(Image.open(self.target_name_list[item],'r'))

        if self.aug:
            check_time = 5
            for i in range(check_time):
                image_t,target_t = self.transform(image,target)
                if np.max(target_t) != np.min(target_t) or i == check_time - 1:
                    image = image_t
                    target = target_t
                    break
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()/255

        targets = []
        for tran in self.multi_scale_transform:
            resize = tran(image=target)['image']
            resize = torch.from_numpy(resize).long()
            targets.append(resize)

        return image, targets

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
