from models.deblurnet import Net
from models.fpn import FPN
import torch
import numpy as np
import cv2
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
import os
import tqdm
import argparse
from cfg import Configs
from torch.nn import DataParallel
import albumentations as albu


class Predictor:
    def __init__(self, weight_path, config):
        assert weight_path != ""
        if config["fpn"]:
            net = FPN.fromConfig(config)
        else:
            net = Net(config)
        net = DataParallel(net)
        params = torch.load(weight_path)
        state_dict = params['model_state_dict']
        net.load_state_dict(state_dict)
        net.eval()
        self.model = net.cuda()
        self.trans = transforms.Compose([
            transforms.ToTensor()])

    def paddingIfNeed(self,img):
        img = np.array(img)
        height, width, _ = img.shape
        padded_height, padded_width,_ = img.shape
        if padded_height % 32 != 0:
            padded_height = (padded_height // 32 + 1) * 32
        if padded_width % 32 != 0:
            padded_width = (padded_width // 32 + 1) * 32
        pad = albu.PadIfNeeded(padded_height,padded_width)
        crop = albu.CenterCrop(height,width)
        img = pad(image=img)["image"]
        return img, crop

    def predict(self, inp: str, target: str = None, merge_img=False):
        assert os.path.exists(inp)

        img = Image.open(inp, 'r')
        img_resize, crop = self.paddingIfNeed(img)
        output, output_view = self.predict_(img_resize)
        img_resize = crop(image=img_resize)["image"]
        output = crop(image=output)["image"]
        output_view = crop(image=output_view)["image"]
        if target:
            tar = cv2.imread(target)
            tar_resize = tar
            tar_resize_view = tar_resize* 127
            return np.hstack((img_resize, tar_resize_view, output_view)) if merge_img else output_view, output, tar_resize
        else:
            return np.hstack(img_resize, output_view) if merge_img else output_view, output, None

    def predict_(self, inp: np.array):
        org = np.array(inp)
        inp = self.trans(inp)
        inp = torch.unsqueeze(inp, 0).cuda()
        output = self.model(inp)[0].cpu().detach().numpy()[0]
        output = np.argmax(output, axis=0)[:, :, np.newaxis].repeat(3, 2)
        return output.astype(np.uint8)[:, :, 0], output.astype(np.uint8)*127

    def predict_dir(self, img_path, tar_path=None, out_dir="./submit2", merge_img=False):
        img_paths = glob(os.path.join(img_path, "*"))

        os.makedirs(os.path.join(out_dir, 'view'),exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'gt'),exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict'),exist_ok=True)
        if tar_path:
            tar_paths = glob(os.path.join(tar_path, "*"))
            assert len(img_paths) == len(tar_paths)

            img_paths.sort()
            tar_paths.sort()
            bar = tqdm.tqdm(zip(img_paths, tar_paths), total=len(img_paths))

            try:
                for img, tar in bar:
                    # print(img,tar)
                    view, output, target = self.predict(img, tar, merge_img=merge_img)

                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(tar)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(tar)), output)
                    cv2.imwrite(os.path.join(out_dir, 'gt', os.path.basename(tar)), target)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()
        else:
            bar = tqdm.tqdm(img_paths, total=len(img_paths))
            try:
                for img in bar:
                    view, output, _ = self.predict(img, merge_img=merge_img)
                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(tar_path)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(tar_path)), output)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, default="./submit")
    parser.add_argument("-m", "--merge_img", type=bool, default=True)
    parser.add_argument("-w", "--weight_path", type=str, default=Configs["model_save_path"])
    args = parser.parse_args()
    predictor = Predictor(args.weight_path, Configs)
    predictor.predict_dir(Configs["test_image_dir"],
                          Configs["test_mask_dir"],
                          out_dir=args.out_dir,
                          merge_img=args.merge_img)
    # predictor = Predictor("mnet.pth")
    # predictor.predict_dir(img_path='C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_image',
    #                       tar_path='C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_gt',
    #                       merge_img=True)
